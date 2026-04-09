//! GPU fault simulation — stuck-at, bridging, delay faults
//!
//! Injects manufacturing defects into digital designs and measures
//! fault coverage via scan chain testing and ATPG-style patterns.

/// Fault types in digital circuits
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FaultType {
    StuckAt0, StuckAt1,
    Bridging,  // short between adjacent nets
    Delay,     // timing violation
    Open,      // broken connection
}

impl std::fmt::Display for FaultType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            FaultType::StuckAt0 => write!(f, "SA0"),
            FaultType::StuckAt1 => write!(f, "SA1"),
            FaultType::Bridging => write!(f, "BRG"),
            FaultType::Delay => write!(f, "DLY"),
            FaultType::Open => write!(f, "OPN"),
        }
    }
}

/// A single fault in the circuit
#[derive(Debug, Clone)]
pub struct Fault {
    pub id: usize,
    pub net: String,
    pub fault_type: FaultType,
    pub detected: bool,
    pub detection_count: u32,
}

/// Test pattern for scan chain
#[derive(Debug, Clone)]
pub struct TestPattern {
    pub inputs: Vec<bool>,
    pub expected_outputs: Vec<bool>,
    pub faults_detected: Vec<usize>,
}

/// Fault simulation result
#[derive(Debug)]
pub struct FaultCoverage {
    pub total_faults: usize,
    pub detected_faults: usize,
    pub coverage_pct: f64,
    pub undetected: Vec<String>,
}

/// Net in a simple combinational circuit
#[derive(Debug, Clone)]
pub struct Net {
    pub name: String,
    pub value: bool,
    pub fanout: usize,  // number of connected gates
}

/// Simple fault simulator for combinational logic
pub struct FaultSimulator {
    pub nets: Vec<Net>,
    pub faults: Vec<Fault>,
    pub patterns: Vec<TestPattern>,
    pub fault_counter: usize,
}

impl FaultSimulator {
    pub fn new() -> Self {
        FaultSimulator {
            nets: Vec::new(), faults: Vec::new(), patterns: Vec::new(), fault_counter: 0,
        }
    }

    /// Add a net to the circuit
    pub fn add_net(&mut self, name: &str, fanout: usize) {
        self.nets.push(Net { name: name.to_string(), value: false, fanout });
    }

    /// Inject all possible stuck-at faults for the circuit
    pub fn inject_stuck_at(&mut self) {
        for net in &self.nets {
            for ft in [FaultType::StuckAt0, FaultType::StuckAt1] {
                self.fault_counter += 1;
                self.faults.push(Fault {
                    id: self.fault_counter, net: net.name.clone(),
                    fault_type: ft, detected: false, detection_count: 0,
                });
            }
        }
    }

    /// Inject bridging faults between adjacent nets
    pub fn inject_bridging(&mut self) {
        for i in 0..self.nets.len().saturating_sub(1) {
            self.fault_counter += 1;
            self.faults.push(Fault {
                id: self.fault_counter,
                net: format!("{}-{}", self.nets[i].name, self.nets[i+1].name),
                fault_type: FaultType::Bridging, detected: false, detection_count: 0,
            });
        }
    }

    /// Generate random test patterns
    pub fn generate_random_patterns(&mut self, n_patterns: usize, n_inputs: usize, seed: u64) {
        let mut rng = seed;
        for _ in 0..n_patterns {
            let mut inputs = Vec::with_capacity(n_inputs);
            let mut expected = Vec::with_capacity(n_inputs);
            for _ in 0..n_inputs {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let val = (rng >> 63) & 1 == 1;
                inputs.push(val);
                expected.push(val); // trivial: output = input for buffer logic
            }
            self.patterns.push(TestPattern {
                inputs, expected_outputs: expected, faults_detected: Vec::new(),
            });
        }
    }

    /// Run scan test simulation
    pub fn run_scan_test(&mut self) -> FaultCoverage {
        // Reset detection
        for fault in &mut self.faults {
            fault.detected = false;
            fault.detection_count = 0;
        }

        // Apply each pattern
        for pattern in &self.patterns {
            // Set net values from pattern inputs
            for (i, net) in self.nets.iter_mut().enumerate() {
                net.value = pattern.inputs.get(i).copied().unwrap_or(false);
            }

            // Check which faults are detected
            for fault in &mut self.faults {
                let expected = self.nets.iter()
                    .find(|n| n.name == fault.net)
                    .map(|n| n.value)
                    .unwrap_or(false);

                let faulty = match fault.fault_type {
                    FaultType::StuckAt0 => false,
                    FaultType::StuckAt1 => true,
                    _ => continue, // skip bridging/delay for now
                };

                if expected != faulty {
                    fault.detected = true;
                    fault.detection_count += 1;
                }
            }
        }

        let detected = self.faults.iter().filter(|f| f.detected).count();
        let total = self.faults.len();
        let undetected: Vec<String> = self.faults.iter()
            .filter(|f| !f.detected)
            .map(|f| format!("{}:{}", f.net, f.fault_type))
            .collect();

        FaultCoverage {
            total_faults: total, detected_faults: detected,
            coverage_pct: if total > 0 { detected as f64 / total as f64 * 100.0 } else { 0.0 },
            undetected,
        }
    }

    /// Compute yield impact of undetected faults
    pub fn yield_impact(&self, coverage: &FaultCoverage, defect_rate: f64) -> f64 {
        // Escaped defects per die
        let escaped = (1.0 - coverage.coverage_pct / 100.0) * defect_rate;
        // Poisson probability of zero escaped defects
        (-escaped).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stuck_at_faults() {
        let mut sim = FaultSimulator::new();
        for i in 0..8 { sim.add_net(&format!("n{}", i), 2); }
        sim.inject_stuck_at();
        assert_eq!(sim.faults.len(), 16); // 8 nets × 2 fault types
    }

    #[test]
    fn test_scan_coverage() {
        let mut sim = FaultSimulator::new();
        for i in 0..8 { sim.add_net(&format!("n{}", i), 2); }
        sim.inject_stuck_at();
        sim.generate_random_patterns(50, 8, 42);
        let coverage = sim.run_scan_test();
        assert!(coverage.coverage_pct > 50.0, "Coverage should be >50%, got {:.1}%", coverage.coverage_pct);
        println!("Scan coverage: {:.1}% ({}/{})", coverage.coverage_pct, coverage.detected_faults, coverage.total_faults);
    }

    #[test]
    fn test_yield_impact() {
        let sim = FaultSimulator::new();
        let coverage = FaultCoverage { total_faults: 100, detected_faults: 95, coverage_pct: 95.0, undetected: vec![] };
        let yield_pct = sim.yield_impact(&coverage, 0.05);
        assert!(yield_pct > 0.99, "High coverage should give high yield");
        println!("Yield impact: {:.4}% at 95% coverage, 0.05 defect rate", yield_pct * 100.0);
    }

    #[test]
    fn test_bridging_faults() {
        let mut sim = FaultSimulator::new();
        for i in 0..4 { sim.add_net(&format!("n{}", i), 1); }
        sim.inject_bridging();
        assert_eq!(sim.faults.len(), 3); // 4-1 adjacent pairs
    }
}
