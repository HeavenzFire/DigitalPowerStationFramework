"""
Scientific Validation - Rigorous validation of mathematical models and physics
Based on established scientific principles and peer-reviewed methodologies
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class ScientificValidator:
    """
    Comprehensive scientific validation of the Digital Power Plant models
    Validates against established physics, mathematics, and engineering principles
    """
    
    def __init__(self):
        self.validation_results = {}
        self.tolerance = 1e-6  # Numerical tolerance for validation
        
    def validate_quantum_coherence_model(self) -> Dict[str, Any]:
        """
        Validate quantum coherence model against theoretical predictions
        
        Returns:
            Validation results with statistical tests
        """
        from .mathematical_foundations import QuantumCoherenceModel
        
        # Theoretical predictions
        alpha = 0.1
        sigma = 0.05
        expected_steady_state = 1.0
        expected_variance = sigma**2 / (2 * alpha)
        
        # Run simulation
        model = QuantumCoherenceModel(alpha=alpha, sigma=sigma)
        n_steps = 10000
        coherence_values = []
        
        for _ in range(n_steps):
            coherence_values.append(model.update())
            
        # Statistical analysis
        mean_coherence = np.mean(coherence_values[-1000:])  # Last 1000 steps
        var_coherence = np.var(coherence_values[-1000:])
        
        # Validation tests
        steady_state_error = abs(mean_coherence - expected_steady_state)
        variance_error = abs(var_coherence - expected_variance)
        
        # Kolmogorov-Smirnov test for distribution
        # OU process should be normally distributed around steady state
        ks_stat, ks_pvalue = stats.kstest(
            coherence_values[-1000:], 
            lambda x: stats.norm.cdf(x, loc=expected_steady_state, scale=np.sqrt(expected_variance))
        )
        
        # Autocorrelation test (OU process should have exponential decay)
        autocorr = np.correlate(coherence_values, coherence_values, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Fit exponential decay: A * exp(-t/tau)
        t = np.arange(len(autocorr))
        try:
            popt, _ = minimize(
                lambda params: np.sum((autocorr - params[0] * np.exp(-t / params[1]))**2),
                x0=[1.0, 10.0],
                method='BFGS'
            ).x, None
            theoretical_tau = 1 / alpha  # Expected time constant
            tau_error = abs(popt[1] - theoretical_tau)
        except:
            tau_error = float('inf')
            
        results = {
            'test_name': 'Quantum Coherence Model',
            'steady_state_error': steady_state_error,
            'variance_error': variance_error,
            'ks_pvalue': ks_pvalue,
            'tau_error': tau_error,
            'passed': (
                steady_state_error < self.tolerance and
                variance_error < self.tolerance and
                ks_pvalue > 0.05 and  # p > 0.05 means normal distribution
                tau_error < 1.0  # Within 1 time constant
            ),
            'mean_coherence': mean_coherence,
            'variance_coherence': var_coherence,
            'expected_steady_state': expected_steady_state,
            'expected_variance': expected_variance
        }
        
        return results
        
    def validate_power_grid_dynamics(self) -> Dict[str, Any]:
        """
        Validate power grid dynamics against swing equation theory
        
        Returns:
            Validation results
        """
        from .mathematical_foundations import PowerGridDynamics
        
        # Test parameters
        M = 1.0  # Inertia
        D = 0.1  # Damping
        K = 1.0  # Synchronizing coefficient
        
        grid = PowerGridDynamics(M=M, D=D, K=K)
        
        # Test step response
        n_steps = 1000
        delta_values = []
        omega_values = []
        
        # Apply step input
        for i in range(n_steps):
            P_m = 1.0 if i > 100 else 0.0  # Step input
            P_e = 0.8  # Constant load
            delta, omega = grid.update(P_m, P_e)
            delta_values.append(delta)
            omega_values.append(omega)
            
        # Analyze response
        final_delta = delta_values[-1]
        final_omega = omega_values[-1]
        
        # For step input, steady state should be: delta = arcsin((P_m - P_e)/K)
        expected_delta = np.arcsin((1.0 - 0.8) / K)
        delta_error = abs(final_delta - expected_delta)
        
        # Frequency should return to base frequency (omega = 0)
        omega_error = abs(final_omega)
        
        # Check stability (frequency within ±0.5 Hz)
        frequency = grid.get_frequency()
        frequency_stable = abs(frequency - 50.0) < 0.5
        
        results = {
            'test_name': 'Power Grid Dynamics',
            'delta_error': delta_error,
            'omega_error': omega_error,
            'frequency_stable': frequency_stable,
            'passed': (
                delta_error < 0.1 and  # Within 0.1 rad
                omega_error < 0.1 and  # Within 0.1 rad/s
                frequency_stable
            ),
            'final_delta': final_delta,
            'final_omega': final_omega,
            'frequency': frequency,
            'expected_delta': expected_delta
        }
        
        return results
        
    def validate_thermodynamic_model(self) -> Dict[str, Any]:
        """
        Validate thermodynamic model against first and second laws
        
        Returns:
            Validation results
        """
        from .mathematical_foundations import ThermodynamicModel
        
        thermo = ThermodynamicModel(T_hot=800.0, T_cold=300.0)
        
        # Test energy conservation (First Law)
        Q_in = 1000.0  # W
        W_out = 400.0  # W
        results = thermo.update(Q_in, W_out)
        
        # Energy balance: Q_in = Q_out + W_out
        energy_balance_error = abs(Q_in - (results['heat_output'] + W_out))
        
        # Carnot efficiency check
        eta_carnot = 1 - thermo.T_cold / thermo.T_hot
        eta_actual = results['efficiency_actual']
        efficiency_valid = eta_actual <= eta_carnot  # Cannot exceed Carnot efficiency
        
        # Entropy increase check (Second Law)
        entropy_increase = results['entropy'] > 0  # Entropy should increase
        
        results_dict = {
            'test_name': 'Thermodynamic Model',
            'energy_balance_error': energy_balance_error,
            'efficiency_valid': efficiency_valid,
            'entropy_increase': entropy_increase,
            'passed': (
                energy_balance_error < 1.0 and  # Within 1 W
                efficiency_valid and
                entropy_increase
            ),
            'eta_carnot': eta_carnot,
            'eta_actual': eta_actual,
            'energy_balance': Q_in - (results['heat_output'] + W_out)
        }
        
        return results_dict
        
    def validate_control_system(self) -> Dict[str, Any]:
        """
        Validate PID control system against control theory
        
        Returns:
            Validation results
        """
        from .mathematical_foundations import ControlSystemTheory
        
        # Test PID controller
        controller = ControlSystemTheory(Kp=1.0, Ki=0.1, Kd=0.01)
        
        # Step response test
        setpoint = 100.0
        n_steps = 1000
        outputs = []
        errors = []
        
        current_value = 0.0
        for i in range(n_steps):
            output = controller.update(setpoint, current_value)
            outputs.append(output)
            errors.append(setpoint - current_value)
            
            # Simple plant model: first-order system
            # dcurrent/dt = (output - current) / tau
            tau = 10.0  # Time constant
            current_value += (output - current_value) / tau * 0.01
            
        # Analyze response
        final_error = abs(errors[-1])
        overshoot = max(outputs) - setpoint if max(outputs) > setpoint else 0
        settling_time = None
        
        # Find settling time (within 2% of final value)
        tolerance = 0.02 * setpoint
        for i, error in enumerate(errors):
            if abs(error) < tolerance:
                settling_time = i * 0.01  # Convert to seconds
                break
                
        # Stability check (no oscillations)
        error_std = np.std(errors[-100:])  # Last 100 steps
        stable = error_std < 1.0  # Low standard deviation
        
        results = {
            'test_name': 'Control System',
            'final_error': final_error,
            'overshoot': overshoot,
            'settling_time': settling_time,
            'stable': stable,
            'passed': (
                final_error < 1.0 and  # Within 1 unit
                overshoot < 10.0 and  # Less than 10% overshoot
                stable
            ),
            'error_std': error_std
        }
        
        return results
        
    def validate_lyapunov_stability(self) -> Dict[str, Any]:
        """
        Validate Lyapunov stability analysis
        
        Returns:
            Validation results
        """
        from .mathematical_foundations import LyapunovStabilityAnalysis
        
        lyapunov = LyapunovStabilityAnalysis()
        
        # Test with known stable system: A = [[-1, 0], [0, -1]]
        A_stable = np.array([[-1.0, 0.0], [0.0, -1.0]])
        stable_result = lyapunov.analyze_stability(A_stable)
        
        # Test with known unstable system: A = [[1, 0], [0, 1]]
        A_unstable = np.array([[1.0, 0.0], [0.0, 1.0]])
        unstable_result = lyapunov.analyze_stability(A_unstable)
        
        # Test with marginally stable system: A = [[0, 1], [-1, 0]]
        A_marginal = np.array([[0.0, 1.0], [-1.0, 0.0]])
        marginal_result = lyapunov.analyze_stability(A_marginal)
        
        results = {
            'test_name': 'Lyapunov Stability Analysis',
            'stable_system_correct': stable_result.get('is_stable', False),
            'unstable_system_correct': not unstable_result.get('is_stable', True),
            'marginal_system_handled': 'is_stable' in marginal_result,
            'passed': (
                stable_result.get('is_stable', False) and
                not unstable_result.get('is_stable', True) and
                'is_stable' in marginal_result
            ),
            'stable_result': stable_result,
            'unstable_result': unstable_result,
            'marginal_result': marginal_result
        }
        
        return results
        
    def validate_quantum_information(self) -> Dict[str, Any]:
        """
        Validate quantum information theory calculations
        
        Returns:
            Validation results
        """
        from .mathematical_foundations import QuantumInformationTheory
        
        quantum_info = QuantumInformationTheory()
        
        # Test fidelity calculation
        # Pure states should have fidelity = 1
        rho_pure1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        rho_pure2 = np.array([[1.0, 0.0], [0.0, 0.0]])
        fidelity_same = quantum_info.calculate_fidelity(rho_pure1, rho_pure2)
        
        # Orthogonal states should have fidelity = 0
        rho_orth1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        rho_orth2 = np.array([[0.0, 0.0], [0.0, 1.0]])
        fidelity_orth = quantum_info.calculate_fidelity(rho_orth1, rho_orth2)
        
        # Test entropy calculation
        # Pure state should have entropy = 0
        rho_pure = np.array([[1.0, 0.0], [0.0, 0.0]])
        entropy_pure = quantum_info.calculate_entanglement_entropy(rho_pure)
        
        # Maximally mixed state should have entropy = 1 (for 1 qubit)
        rho_mixed = np.array([[0.5, 0.0], [0.0, 0.5]])
        entropy_mixed = quantum_info.calculate_entanglement_entropy(rho_mixed)
        
        results = {
            'test_name': 'Quantum Information Theory',
            'fidelity_same': fidelity_same,
            'fidelity_orth': fidelity_orth,
            'entropy_pure': entropy_pure,
            'entropy_mixed': entropy_mixed,
            'passed': (
                abs(fidelity_same - 1.0) < self.tolerance and
                abs(fidelity_orth) < self.tolerance and
                abs(entropy_pure) < self.tolerance and
                abs(entropy_mixed - 1.0) < self.tolerance
            )
        }
        
        return results
        
    def validate_statistical_mechanics(self) -> Dict[str, Any]:
        """
        Validate statistical mechanics calculations
        
        Returns:
            Validation results
        """
        from .mathematical_foundations import StatisticalMechanics
        
        stats_mech = StatisticalMechanics(T=300.0)
        
        # Test Boltzmann distribution
        energies = np.array([0.0, 1.0, 2.0, 3.0]) * 1.6e-19  # eV to J
        probabilities = stats_mech.boltzmann_distribution(energies)
        
        # Probabilities should sum to 1
        prob_sum = np.sum(probabilities)
        prob_sum_error = abs(prob_sum - 1.0)
        
        # Lower energy states should have higher probability
        prob_decreasing = all(probabilities[i] >= probabilities[i+1] for i in range(len(probabilities)-1))
        
        # Test entropy calculation
        entropy = stats_mech.calculate_entropy(probabilities)
        entropy_valid = entropy >= 0  # Entropy should be non-negative
        
        # Test average energy calculation
        avg_energy = stats_mech.calculate_average_energy(energies, probabilities)
        avg_energy_valid = avg_energy >= 0  # Average energy should be non-negative
        
        results = {
            'test_name': 'Statistical Mechanics',
            'prob_sum_error': prob_sum_error,
            'prob_decreasing': prob_decreasing,
            'entropy_valid': entropy_valid,
            'avg_energy_valid': avg_energy_valid,
            'passed': (
                prob_sum_error < self.tolerance and
                prob_decreasing and
                entropy_valid and
                avg_energy_valid
            ),
            'prob_sum': prob_sum,
            'entropy': entropy,
            'avg_energy': avg_energy
        }
        
        return results
        
    def run_all_validations(self) -> Dict[str, Any]:
        """
        Run all scientific validations
        
        Returns:
            Comprehensive validation results
        """
        logger.info("🔬 Starting comprehensive scientific validation...")
        
        validations = [
            self.validate_quantum_coherence_model,
            self.validate_power_grid_dynamics,
            self.validate_thermodynamic_model,
            self.validate_control_system,
            self.validate_lyapunov_stability,
            self.validate_quantum_information,
            self.validate_statistical_mechanics
        ]
        
        results = {}
        passed_tests = 0
        total_tests = len(validations)
        
        for validation_func in validations:
            try:
                result = validation_func()
                results[result['test_name']] = result
                if result['passed']:
                    passed_tests += 1
                    logger.info(f"✅ {result['test_name']}: PASSED")
                else:
                    logger.warning(f"❌ {result['test_name']}: FAILED")
            except Exception as e:
                logger.error(f"💥 {validation_func.__name__}: ERROR - {e}")
                results[validation_func.__name__] = {
                    'test_name': validation_func.__name__,
                    'passed': False,
                    'error': str(e)
                }
                
        # Overall validation result
        overall_passed = passed_tests == total_tests
        success_rate = passed_tests / total_tests * 100
        
        results['overall'] = {
            'passed': overall_passed,
            'success_rate': success_rate,
            'passed_tests': passed_tests,
            'total_tests': total_tests
        }
        
        logger.info(f"🎯 Validation Complete: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        if overall_passed:
            logger.info("🎉 All scientific validations PASSED! System is mathematically sound.")
        else:
            logger.warning("⚠️ Some validations FAILED. Review results for details.")
            
        return results
        
    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive validation report
        
        Args:
            results: Validation results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("DIGITAL POWER PLANT - SCIENTIFIC VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall summary
        overall = results.get('overall', {})
        report.append(f"OVERALL RESULT: {'PASSED' if overall.get('passed', False) else 'FAILED'}")
        report.append(f"Success Rate: {overall.get('success_rate', 0):.1f}%")
        report.append(f"Tests Passed: {overall.get('passed_tests', 0)}/{overall.get('total_tests', 0)}")
        report.append("")
        
        # Individual test results
        report.append("DETAILED RESULTS:")
        report.append("-" * 40)
        
        for test_name, result in results.items():
            if test_name == 'overall':
                continue
                
            status = "PASSED" if result.get('passed', False) else "FAILED"
            report.append(f"{test_name}: {status}")
            
            if not result.get('passed', False) and 'error' not in result:
                # Show specific failure details
                for key, value in result.items():
                    if key not in ['test_name', 'passed'] and isinstance(value, (int, float)):
                        report.append(f"  {key}: {value:.6f}")
            elif 'error' in result:
                report.append(f"  Error: {result['error']}")
                
            report.append("")
            
        report.append("=" * 80)
        report.append("Validation completed based on rigorous mathematical foundations")
        report.append("including quantum mechanics, thermodynamics, control theory,")
        report.append("and statistical mechanics principles.")
        report.append("=" * 80)
        
        return "\n".join(report)

# Example usage
def run_scientific_validation():
    """Run complete scientific validation and generate report"""
    validator = ScientificValidator()
    results = validator.run_all_validations()
    report = validator.generate_validation_report(results)
    
    print(report)
    
    # Save report to file
    with open('/workspace/digital_power_plant/validation_report.txt', 'w') as f:
        f.write(report)
        
    return results

if __name__ == "__main__":
    run_scientific_validation()