#!/usr/bin/env python3
"""
Scientific Validation Script for Digital Power Plant
Run this script to validate all mathematical models and physical principles
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('validation.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run comprehensive scientific validation"""
    
    print("🔬 DIGITAL POWER PLANT - SCIENTIFIC VALIDATION")
    print("=" * 60)
    print("Validating mathematical models and physical principles...")
    print()
    
    try:
        # Import validation modules
        from core.scientific_validation import ScientificValidator
        from core.mathematical_foundations import validate_mathematical_models
        
        # Run mathematical model validation
        print("📐 Validating Mathematical Models...")
        validate_mathematical_models()
        print()
        
        # Run comprehensive scientific validation
        print("🧪 Running Comprehensive Scientific Validation...")
        validator = ScientificValidator()
        results = validator.run_all_validations()
        
        # Generate and display report
        print("\n" + "=" * 60)
        report = validator.generate_validation_report(results)
        print(report)
        
        # Save detailed results
        import json
        with open('validation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📊 Detailed results saved to: validation_results.json")
        print(f"📝 Validation log saved to: validation.log")
        
        # Return success/failure status
        overall_passed = results.get('overall', {}).get('passed', False)
        if overall_passed:
            print("\n🎉 ALL VALIDATIONS PASSED! System is scientifically sound.")
            return 0
        else:
            print("\n⚠️ SOME VALIDATIONS FAILED! Review results for details.")
            return 1
            
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"\n❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        return 1
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        print(f"\n💥 Validation error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)