//! GPU yield-aware MoE swarm tiling — Monte Carlo die simulation
//!
//! Simulates wafer fabrication defects using Poisson distribution,
//! grades dies (GOLD/SILVER/BRONZE/SCRAP), and routes MoE experts
//! to surviving tiles.

use crate::VesselSpec;
use std::collections::HashMap;

/// Die grade based on defect analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DieGrade {
    Gold,    // Center zone, 0 defects
    Silver,  // Mid zone, minor defects repairable
    Bronze,  // Outer zone, degraded but functional
    Scrap,   // Beyond repair
}

impl DieGrade {
    pub fn performance_factor(&self) -> f64 {
        match self {
            DieGrade::Gold => 1.0,
            DieGrade::Silver => 0.85,
            DieGrade::Bronze => 0.65,
            DieGrade::Scrap => 0.0,
        }
    }
}

/// Single die on wafer
#[derive(Debug, Clone)]
pub struct Die {
    pub x: usize, pub y: usize,
    pub grade: DieGrade,
    pub defects: u32,
    pub is_functional: bool,
}

/// Yield simulation result
#[derive(Debug)]
pub struct YieldResult {
    pub total_dies: u32,
    pub gold: u32, pub silver: u32, pub bronze: u32, pub scrap: u32,
    pub gold_pct: f64, pub silver_pct: f64, pub bronze_pct: f64, pub scrap_pct: f64,
    pub avg_defects: f64,
}

/// Circular mask zone for defect probability
#[derive(Debug, Clone)]
pub struct WaferMask {
    pub wafer_mm: f64,   // wafer diameter
    pub exclusion_mm: f64, // edge exclusion zone
}

impl Default for WaferMask {
    fn default() -> Self { WaferMask { wafer_mm: 300.0, exclusion_mm: 3.0 } }
}

/// Swarm tiler with Monte Carlo yield simulation
pub struct SwarmTiler {
    pub die_size_mm: f64,
    pub defect_rate: f64,     // defects per cm²
    pub mask: WaferMask,
    pub repairable_defects: u32,
}

impl SwarmTiler {
    pub fn new(die_size_mm: f64, defect_rate: f64) -> Self {
        SwarmTiler {
            die_size_mm,
            defect_rate,
            mask: WaferMask::default(),
            repairable_defects: 2,
        }
    }

    /// Place dies on wafer and simulate defects
    pub fn simulate_wafer(&self, seed: u64) -> (Vec<Die>, f64) {
        let mut dies = Vec::new();
        let r = self.mask.wafer_mm / 2.0;
        let n = (r / self.die_size_mm) as usize;
        let center = n as f64 / 2.0;
        let die_area_cm2 = (self.die_size_mm / 10.0).powi(2);

        // Simple LCG for reproducibility
        let mut rng = seed;

        for yi in 0..n {
            for xi in 0..n {
                // Check if die center is within wafer circle
                let cx = xi as f64 + 0.5 - center;
                let cy = yi as f64 + 0.5 - center;
                let dist = (cx * cy).sqrt().abs().max((cx * cx + cy * cy).sqrt() * 0.01);
                let actual_dist = (cx * cx + cy * cy).sqrt() * self.die_size_mm;

                if actual_dist + self.die_size_mm / 2.0 > r {
                    continue; // outside wafer
                }

                // Poisson-distributed defects (Knuth algorithm)
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u01 = ((rng >> 33) as f64) / (u64::MAX as f64 / 2.0 + 1.0);
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u02 = ((rng >> 33) as f64) / (u64::MAX as f64 / 2.0 + 1.0);
                let expected = self.defect_rate * die_area_cm2;
                let mut defects = 0u32;
                let mut l = (-expected).exp();
                let mut p = 1.0;
                loop {
                    p *= u02;
                    if p < l { break; }
                    defects += 1;
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let un = ((rng >> 33) as f64) / (u64::MAX as f64 / 2.0 + 1.0);
                    p *= un;
                }

                // Zone-based grading (circular mask)
                let zone = actual_dist / r; // 0=center, 1=edge
                let grade = if defects == 0 {
                    DieGrade::Gold
                } else if defects <= self.repairable_defects && zone < 0.85 {
                    DieGrade::Silver
                } else if defects <= self.repairable_defects + 1 && zone < 0.7 {
                    DieGrade::Bronze
                } else {
                    DieGrade::Scrap
                };

                dies.push(Die {
                    x: xi, y: yi,
                    grade: grade.clone(),
                    defects,
                    is_functional: grade != DieGrade::Scrap,
                });
            }
        }

        let dies_per_mm2 = 1.0 / self.die_size_mm.powi(2);
        let wafer_area = std::f64::consts::PI * r.powi(2);
        (dies, wafer_area * dies_per_mm2)
    }

    /// Run yield simulation and compute statistics
    pub fn yield_analysis(&self, trials: u32) -> YieldResult {
        let mut total_gold = 0u32; let mut total_silver = 0u32;
        let mut total_bronze = 0u32; let mut total_scrap = 0u32;
        let mut total_dies = 0u32; let mut total_defects = 0u64;

        for trial in 0..trials {
            let (dies, _) = self.simulate_wafer(trial as u64);
            total_dies += dies.len() as u32;
            for d in &dies {
                total_defects += d.defects as u64;
                match d.grade {
                    DieGrade::Gold => total_gold += 1,
                    DieGrade::Silver => total_silver += 1,
                    DieGrade::Bronze => total_bronze += 1,
                    DieGrade::Scrap => total_scrap += 1,
                }
            }
        }

        let t = total_dies as f64;
        YieldResult {
            total_dies,
            gold: total_gold, silver: total_silver,
            bronze: total_bronze, scrap: total_scrap,
            gold_pct: total_gold as f64 / t * 100.0,
            silver_pct: total_silver as f64 / t * 100.0,
            bronze_pct: total_bronze as f64 / t * 100.0,
            scrap_pct: total_scrap as f64 / t * 100.0,
            avg_defects: total_defects as f64 / t,
        }
    }

    /// Estimate unit cost given yield
    pub fn unit_cost(&self, yield_result: &YieldResult, wafer_cost: f64) -> HashMap<DieGrade, f64> {
        let functional = yield_result.gold + yield_result.silver + yield_result.bronze;
        let cost_per_die = if functional > 0 {
            wafer_cost / functional as f64
        } else {
            f64::MAX
        };
        let mut costs = HashMap::new();
        costs.insert(DieGrade::Gold, cost_per_die);
        costs.insert(DieGrade::Silver, cost_per_die / DieGrade::Silver.performance_factor());
        costs.insert(DieGrade::Bronze, cost_per_die / DieGrade::Bronze.performance_factor());
        costs.insert(DieGrade::Scrap, 0.0);
        costs
    }

    /// MoE expert routing — assign experts to best-available dies
    pub fn route_experts(&self, n_experts: u32, dies: &[Die]) -> HashMap<DieGrade, u32> {
        let functional: Vec<&Die> = dies.iter().filter(|d| d.is_functional).collect();
        let mut assignment = HashMap::new();
        assignment.insert(DieGrade::Gold, 0);
        assignment.insert(DieGrade::Silver, 0);
        assignment.insert(DieGrade::Bronze, 0);
        assignment.insert(DieGrade::Scrap, 0);

        // Prefer gold dies first
        let gold = functional.iter().filter(|d| d.grade == DieGrade::Gold).count() as u32;
        let silver = functional.iter().filter(|d| d.grade == DieGrade::Silver).count() as u32;
        let bronze = functional.iter().filter(|d| d.grade == DieGrade::Bronze).count() as u32;

        let mut remaining = n_experts;
        *assignment.get_mut(&DieGrade::Gold).unwrap() = remaining.min(gold);
        remaining = remaining.saturating_sub(gold);
        *assignment.get_mut(&DieGrade::Silver).unwrap() = remaining.min(silver);
        remaining = remaining.saturating_sub(silver);
        *assignment.get_mut(&DieGrade::Bronze).unwrap() = remaining.min(bronze);
        assignment
    }
}

/// Estimate die size for a vessel class (28nm INT4)
pub fn estimate_die_size(vessel: &VesselSpec) -> f64 {
    // Rule of thumb: 28nm INT4 ~ 0.2 mm² per billion params
    let core_area = vessel.params_b as f64 * 0.2; // mm²
    // Add 40% for I/O pads, control logic, SRAM
    core_area * 1.4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiler_basic() {
        let tiler = SwarmTiler::new(20.0, 0.05); // 20mm die, 0.05 defects/cm²
        let (dies, max_dies) = tiler.simulate_wafer(42);
        assert!(dies.len() > 50, "Expected many dies, got {}", dies.len());
        let functional = dies.iter().filter(|d| d.is_functional).count();
        assert!(functional > 40, "Expected many functional dies, got {}", functional);
    }

    #[test]
    fn test_yield_analysis() {
        let tiler = SwarmTiler::new(20.0, 0.05);
        let result = tiler.yield_analysis(10);
        assert!(result.gold_pct > 0.0);
        assert!(result.total_dies > 0);
        println!("Yield: Gold={:.1}% Silver={:.1}% Bronze={:.1}% Scrap={:.1}%",
            result.gold_pct, result.silver_pct, result.bronze_pct, result.scrap_pct);
    }

    #[test]
    fn test_die_size_estimate() {
        let size = estimate_die_size(&crate::SCOUT);
        assert!(size > 0.0, "Die size should be positive");
        assert!(size < 1000.0, "Die size should be reasonable mm²");
    }

    #[test]
    fn test_expert_routing() {
        let tiler = SwarmTiler::new(20.0, 0.05);
        let (dies, _) = tiler.simulate_wafer(42);
        let routing = tiler.route_experts(10, &dies);
        let total: u32 = routing.values().sum();
        assert!(total <= 10);
    }
}
