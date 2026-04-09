//! GPU thermal simulation — 2D finite-difference heat diffusion
//!
//! Simulates chip thermal behavior with configurable power sources,
//! thermal vias, and boundary conditions. Designed for mask-locked
//! inference chip floorplan validation.

/// Grid cell in thermal simulation
#[derive(Debug, Clone, Copy)]
pub struct ThermalCell {
    pub temp: f64,        // Celsius
    pub power_mw: f64,    // mW generated at this cell
    pub is_via: bool,     // thermal via present
    pub is_pad: bool,     // I/O pad (fixed boundary)
}

/// Thermal simulation result
#[derive(Debug)]
pub struct ThermalResult {
    pub peak_temp: f64,
    pub avg_temp: f64,
    pub min_temp: f64,
    pub iterations: u32,
    pub converged: bool,
}

/// 2D thermal simulator using finite-difference method
pub struct ThermalSimulator {
    pub grid: Vec<Vec<ThermalCell>>,
    pub ambient: f64,         // ambient temperature (°C)
    pub thermal_conductivity: f64, // W/(mm·K) silicon ~ 150
    pub cell_size_mm: f64,    // grid cell physical size
    pub max_iterations: u32,
    pub convergence_threshold: f64,
}

impl ThermalSimulator {
    /// Create NxN thermal grid
    pub fn new(size: usize, ambient: f64) -> Self {
        let grid = vec![vec![ThermalCell { temp: ambient, power_mw: 0.0, is_via: false, is_pad: false }; size]; size];
        ThermalSimulator {
            grid, ambient,
            thermal_conductivity: 150.0,
            cell_size_mm: 0.1,   // 100μm cells
            max_iterations: 10000,
            convergence_threshold: 0.01,
        }
    }

    /// Add a rectangular power source
    pub fn add_power_source(&mut self, x: usize, y: usize, w: usize, h: usize, power_mw: f64) {
        for yi in y..(y + h).min(self.grid.len()) {
            for xi in x..(x + w).min(self.grid[0].len()) {
                self.grid[yi][xi].power_mw = power_mw / (w * h) as f64;
            }
        }
    }

    /// Add thermal via (enhanced heat conduction)
    pub fn add_thermal_via(&mut self, x: usize, y: usize) {
        if y < self.grid.len() && x < self.grid[0].len() {
            self.grid[y][x].is_via = true;
        }
    }

    /// Add row of I/O pads (fixed to ambient + 10°C)
    pub fn add_pad_row(&mut self, y: usize) {
        if y < self.grid.len() {
            for cell in &mut self.grid[y] {
                cell.is_pad = true;
                cell.temp = self.ambient + 10.0;
            }
        }
    }

    /// Run thermal simulation
    pub fn simulate(&mut self) -> ThermalResult {
        let n = self.grid.len();
        if n == 0 { return ThermalResult { peak_temp: self.ambient, avg_temp: self.ambient, min_temp: self.ambient, iterations: 0, converged: true }; }

        let alpha = self.thermal_conductivity * self.cell_size_mm;
        let dt = 0.01; // stability criterion
        let mut max_delta = f64::MAX;
        let mut iterations = 0u32;

        for _ in 0..self.max_iterations {
            let mut new_temps = vec![vec![0.0f64; n]; n];
            max_delta = 0.0;

            for y in 0..n {
                for x in 0..n {
                    let cell = self.grid[y][x];

                    if cell.is_pad {
                        new_temps[y][x] = cell.temp; // fixed boundary
                        continue;
                    }

                    // Heat generation
                    let heat_in = cell.power_mw * 1e-3; // mW to W

                    // Thermal via: 10x conductivity
                    let k = if cell.is_via { alpha * 10.0 } else { alpha };

                    // Finite difference: 4-neighbor average
                    let mut neighbor_sum = 0.0;
                    let mut neighbor_count = 0;
                    if x > 0 { neighbor_sum += self.grid[y][x-1].temp; neighbor_count += 1; }
                    if x < n-1 { neighbor_sum += self.grid[y][x+1].temp; neighbor_count += 1; }
                    if y > 0 { neighbor_sum += self.grid[y-1][x].temp; neighbor_count += 1; }
                    if y < n-1 { neighbor_sum += self.grid[y+1][x].temp; neighbor_count += 1; }

                    let avg_neighbor = if neighbor_count > 0 { neighbor_sum / neighbor_count as f64 } else { self.ambient };

                    // Diffusion + heat generation
                    let diffusion = k * dt * (avg_neighbor - cell.temp);
                    let new_temp = cell.temp + diffusion + heat_in * dt * 100.0; // scaling factor

                    // Clamp to ambient
                    let new_temp = new_temp.max(self.ambient);
                    new_temps[y][x] = new_temp;

                    let delta = (new_temp - cell.temp).abs();
                    if delta > max_delta { max_delta = delta; }
                }
            }

            // Apply new temperatures
            for y in 0..n {
                for x in 0..n {
                    self.grid[y][x].temp = new_temps[y][x];
                }
            }

            iterations += 1;
            if max_delta < self.convergence_threshold { break; }
        }

        let temps: Vec<f64> = self.grid.iter().flat_map(|row| row.iter().map(|c| c.temp)).collect();
        let peak = temps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = temps.iter().cloned().fold(f64::INFINITY, f64::min);
        let avg = temps.iter().sum::<f64>() / temps.len() as f64;

        ThermalResult {
            peak_temp: peak, avg_temp: avg, min_temp: min,
            iterations, converged: max_delta < self.convergence_threshold,
        }
    }

    /// Get ASCII thermal map
    pub fn thermal_map(&self) -> String {
        let mut map = String::new();
        for row in &self.grid {
            for cell in row {
                let c = match cell.temp {
                    t if t < self.ambient + 5.0 => '.',
                    t if t < self.ambient + 15.0 => '░',
                    t if t < self.ambient + 25.0 => '▒',
                    t if t < self.ambient + 35.0 => '▓',
                    _ => '█',
                };
                map.push(c);
            }
            map.push('\n');
        }
        map
    }

    /// Find hot spots above threshold
    pub fn find_hotspots(&self, threshold: f64) -> Vec<(usize, usize, f64)> {
        let mut spots = Vec::new();
        for y in 0..self.grid.len() {
            for x in 0..self.grid[0].len() {
                if self.grid[y][x].temp > threshold {
                    spots.push((x, y, self.grid[y][x].temp));
                }
            }
        }
        spots.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        spots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_thermal() {
        let mut sim = ThermalSimulator::new(20, 25.0);
        sim.add_power_source(8, 8, 4, 4, 1000.0); // 1W core
        sim.add_pad_row(0);
        sim.add_pad_row(19);
        let result = sim.simulate();
        assert!(result.peak_temp > 25.0, "Core should heat up");
        assert!(result.peak_temp < 200.0, "Should not exceed 200°C");
        println!("Thermal: peak={:.1f}°C avg={:.1f}°C iters={}",
            result.peak_temp, result.avg_temp, result.iterations);
    }

    #[test]
    fn test_thermal_vias() {
        let mut sim = ThermalSimulator::new(20, 25.0);
        sim.add_power_source(8, 8, 4, 4, 1000.0);
        for y in 5..15 { sim.add_thermal_via(y, 5); } // via column
        let result = sim.simulate();
        assert!(result.peak_temp > 25.0);
    }

    #[test]
    fn test_thermal_map() {
        let mut sim = ThermalSimulator::new(10, 25.0);
        sim.add_power_source(4, 4, 2, 2, 500.0);
        sim.simulate();
        let map = sim.thermal_map();
        assert!(map.contains('░') || map.contains('▒'), "Map should show thermal gradient");
    }

    #[test]
    fn test_hotspots() {
        let mut sim = ThermalSimulator::new(20, 25.0);
        sim.add_power_source(8, 8, 4, 4, 5000.0);
        sim.simulate();
        let spots = sim.find_hotspots(50.0);
        assert!(spots.len() > 0, "Should find hotspots");
    }
}
