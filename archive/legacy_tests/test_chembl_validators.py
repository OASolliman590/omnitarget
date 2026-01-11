#!/usr/bin/env python3
"""
Test ChEMBL Validators
Validates all 10 ChEMBL-specific validation methods.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.core.chembl_validation import ChEMBLValidator
from src.models.data_models import Compound, Bioactivity, DrugInfo


def test_validators():
    """Test all 10 ChEMBL validators."""

    print("=" * 80)
    print("ChEMBL Validators Test")
    print("=" * 80)

    validator = ChEMBLValidator()

    # Test 1: Drug-Likeness Validator
    print("\n1. Testing Drug-Likeness Validator...")
    aspirin = Compound(
        chembl_id='CHEMBL25',
        name='ASPIRIN',
        molecular_weight=180.16,
        alogp=1.19,
        hba=4,
        hbd=1,
        psa=63.6,
        rtb=3,
        ro5_violations=0
    )
    valid, msg = validator.validate_drug_likeness(aspirin)
    assert valid, f"Aspirin should be drug-like: {msg}"
    print(f"   ✅ {msg}")

    # Test non-drug-like compound
    large_molecule = Compound(
        chembl_id='CHEMBL999',
        name='LARGE_MOLECULE',
        molecular_weight=650,  # Too heavy
        alogp=6.5,  # Too lipophilic
        hba=12,  # Too many acceptors
        hbd=6,   # Too many donors
        psa=150,
        rtb=15,
        ro5_violations=4
    )
    valid, msg = validator.validate_drug_likeness(large_molecule)
    assert not valid, f"Large molecule should not be drug-like: {msg}"
    print(f"   ✅ Non-drug-like correctly identified: {msg}")

    # Test 2: Bioactivity Quality Validator
    print("\n2. Testing Bioactivity Quality Validator...")
    good_bioactivity = Bioactivity(
        activity_id='12345',
        assay_chembl_id='CHEMBL123',
        target_chembl_id='CHEMBL2095173',
        molecule_chembl_id='CHEMBL25',
        activity_type='IC50',
        activity_value=1500.0,
        activity_units='nM',
        activity_relation='=',
        assay_type='B',
        confidence=0.8
    )
    valid, msg = validator.validate_bioactivity_quality(good_bioactivity)
    assert valid, f"Good bioactivity should pass: {msg}"
    print(f"   ✅ {msg}")

    # Test 3: Bioactivity Range Validator
    print("\n3. Testing Bioactivity Range Validator...")
    valid, msg = validator.validate_bioactivity_range(good_bioactivity)
    assert valid, f"IC50=1500 nM should be in valid range: {msg}"
    print(f"   ✅ {msg}")

    # Test out-of-range value
    bad_bioactivity = Bioactivity(
        activity_id='99999',
        assay_chembl_id='CHEMBL999',
        target_chembl_id='CHEMBL999',
        molecule_chembl_id='CHEMBL999',
        activity_type='IC50',
        activity_value=10_000_000.0,  # 10 mM - too high
        activity_units='nM',
        activity_relation='=',
        assay_type='B',
        confidence=0.7
    )
    valid, msg = validator.validate_bioactivity_range(bad_bioactivity)
    assert not valid, f"IC50=10,000,000 nM should be out of range: {msg}"
    print(f"   ✅ Out-of-range correctly detected: {msg}")

    # Test 4: Drug-Target Association Validator
    print("\n4. Testing Drug-Target Association Validator...")
    drug = DrugInfo(
        drug_id='D00109',
        name='ASPIRIN',
        indication='Pain relief',
        mechanism='COX inhibitor',
        targets=['PTGS1', 'PTGS2'],
        development_status='approved',
        drug_class='NSAID',
        approval_status='approved'
    )
    valid, msg = validator.validate_drug_target_association(drug, 'PTGS1')
    assert valid, f"ASPIRIN-PTGS1 association should be valid: {msg}"
    print(f"   ✅ {msg}")

    # Test invalid association
    valid, msg = validator.validate_drug_target_association(drug, 'EGFR')
    assert not valid, f"ASPIRIN-EGFR association should be invalid: {msg}"
    print(f"   ✅ Invalid association correctly detected: {msg}")

    # Test 5: Compound Structure Validator
    print("\n5. Testing Compound Structure Validator...")
    aspirin_with_structure = Compound(
        chembl_id='CHEMBL25',
        name='ASPIRIN',
        molecular_weight=180.16,
        alogp=1.19,
        hba=4,
        hbd=1,
        psa=63.6,
        rtb=3,
        ro5_violations=0,
        smiles='CC(=O)Oc1ccccc1C(=O)O',
        inchi_key='BSYNRYMUTXBXSQ-UHFFFAOYSA-N'
    )
    valid, msg = validator.validate_compound_structure(aspirin_with_structure)
    assert valid, f"Aspirin structure should be valid: {msg}"
    print(f"   ✅ {msg}")

    # Test 6: Bioactivity Confidence Validator
    print("\n6. Testing Bioactivity Confidence Validator...")
    high_conf = Bioactivity(
        activity_id='12345',
        assay_chembl_id='CHEMBL123',
        target_chembl_id='CHEMBL2095173',
        molecule_chembl_id='CHEMBL25',
        activity_type='IC50',
        activity_value=100.0,
        activity_units='nM',
        activity_relation='=',
        assay_type='B',  # Binding assay
        confidence=0.85  # High confidence
    )
    valid, msg = validator.validate_bioactivity_confidence(high_conf)
    assert valid, f"High confidence bioactivity should pass: {msg}"
    print(f"   ✅ {msg}")

    low_conf = Bioactivity(
        activity_id='99999',
        assay_chembl_id='CHEMBL999',
        target_chembl_id='CHEMBL999',
        molecule_chembl_id='CHEMBL999',
        activity_type='IC50',
        activity_value=1000.0,
        activity_units='nM',
        activity_relation='<',  # Approximate
        assay_type='F',  # Functional assay
        confidence=0.45  # Low confidence
    )
    valid, msg = validator.validate_bioactivity_confidence(low_conf)
    assert not valid, f"Low confidence bioactivity should fail: {msg}"
    print(f"   ✅ Low confidence correctly detected: {msg}")

    # Test 7: Target Druggability Validator
    print("\n7. Testing Target Druggability Validator...")
    bioactivity_data = {
        'activity_count': 150,
        'median_ic50': 50.0
    }
    valid, msg = validator.validate_target_druggability('EGFR', 0.75, bioactivity_data)
    assert valid, f"EGFR druggability should be valid: {msg}"
    print(f"   ✅ {msg}")

    # Test low druggability
    valid, msg = validator.validate_target_druggability('UNKNOWN', 0.15)
    assert not valid, f"Low druggability should fail: {msg}"
    print(f"   ✅ Low druggability correctly detected: {msg}")

    # Test 8: Cross-Database Concordance Validator
    print("\n8. Testing Cross-Database Concordance Validator...")
    kegg_drugs = [
        DrugInfo(drug_id='D001', name='GEFITINIB', targets=['EGFR']),
        DrugInfo(drug_id='D002', name='ERLOTINIB', targets=['EGFR']),
    ]
    chembl_compounds = [
        Compound(chembl_id='CHEMBL1', name='GEFITINIB', molecular_weight=446.9, alogp=4.15, hba=7, hbd=1, psa=68.7, rtb=8, ro5_violations=0),
        Compound(chembl_id='CHEMBL2', name='LAPATINIB', molecular_weight=581.1, alogp=5.5, hba=8, hbd=2, psa=106.4, rtb=10, ro5_violations=1),
    ]
    valid, msg = validator.validate_kegg_chembl_concordance(kegg_drugs, chembl_compounds, 'EGFR')
    assert valid, f"KEGG-ChEMBL concordance should be valid: {msg}"
    print(f"   ✅ {msg}")

    # Test 9: Drug Development Status Validator
    print("\n9. Testing Drug Development Status Validator...")
    approved_drug = DrugInfo(
        drug_id='D00109',
        name='ASPIRIN',
        development_status='approved',
        approval_status='approved',
        targets=['PTGS1', 'PTGS2']
    )
    valid, msg = validator.validate_drug_development_status(approved_drug)
    assert valid, f"Approved drug status should be valid: {msg}"
    print(f"   ✅ {msg}")

    # Test 10: Molecular Descriptor Validator
    print("\n10. Testing Molecular Descriptor Validator...")
    valid, msg = validator.validate_molecular_descriptors(aspirin)
    assert valid, f"Aspirin descriptors should be valid: {msg}"
    print(f"   ✅ {msg}")

    # Test unusual descriptors
    unusual = Compound(
        chembl_id='CHEMBL999',
        name='UNUSUAL',
        molecular_weight=50,  # Too small
        alogp=15,  # Too high
        hba=5,
        hbd=2,
        psa=300,  # Too high
        rtb=25,  # Too many
        ro5_violations=4
    )
    valid, msg = validator.validate_molecular_descriptors(unusual)
    assert not valid, f"Unusual descriptors should fail: {msg}"
    print(f"   ✅ Unusual descriptors correctly detected: {msg}")

    # Test 11: Comprehensive Validation
    print("\n11. Testing Comprehensive Validation...")
    compounds = [aspirin, aspirin_with_structure, large_molecule]
    bioactivities = [good_bioactivity, bad_bioactivity, high_conf, low_conf]
    drugs = [drug, approved_drug]

    results = validator.validate_chembl_data_quality(compounds, bioactivities, drugs)

    print(f"\n   Validation Results:")
    print(f"   Overall Quality Score: {results['overall_quality_score']:.1%}")
    print(f"   Summary: {results['summary']}")
    print(f"\n   Pass Rates:")
    for check, rate in results['pass_rates'].items():
        print(f"     - {check}: {rate:.1%}")

    assert results['overall_quality_score'] >= 0.5, "Overall quality should be acceptable"
    print(f"\n   ✅ Comprehensive validation completed")

    print("\n" + "=" * 80)
    print("✅ ALL 10 CHEMBL VALIDATORS PASSED!")
    print("=" * 80)
    print("\nPhase 5 ChEMBL validation framework is working correctly!")

    return True


if __name__ == "__main__":
    success = test_validators()
    sys.exit(0 if success else 1)
