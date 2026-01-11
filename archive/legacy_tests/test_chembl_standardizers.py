#!/usr/bin/env python3
"""
Test ChEMBL standardization methods.
Validates that all 8 standardization methods work correctly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.core.data_standardizer import DataStandardizer
from src.models.data_models import Compound, Bioactivity, DrugInfo

def test_standardizers():
    """Test ChEMBL standardization methods."""

    print("=" * 80)
    print("ChEMBL Standardization Test")
    print("=" * 80)

    standardizer = DataStandardizer()

    # Test 1: standardize_chembl_compound
    print("\n1. Testing standardize_chembl_compound...")
    compound_raw = {
        'molecule_chembl_id': 'CHEMBL25',
        'pref_name': 'ASPIRIN',
        'molecule_structures': {
            'canonical_smiles': 'CC(=O)Oc1ccccc1C(=O)O',
            'standard_inchi_key': 'BSYNRYMUTXBXSQ-UHFFFAOYSA-N'
        },
        'molecule_properties': {
            'molecular_weight': 180.16,
            'alogp': 1.19,
            'hba': 4,
            'hbd': 1,
            'psa': 63.6,
            'rtb': 3,
            'num_ro5_violations': 0
        }
    }
    compound = standardizer.standardize_chembl_compound(compound_raw)
    assert compound.chembl_id == 'CHEMBL25'
    assert compound.name == 'ASPIRIN'
    assert compound.molecular_weight == 180.16
    print(f"   ✅ Compound: {compound.name} (MW: {compound.molecular_weight})")

    # Test 2: standardize_chembl_bioactivity
    print("\n2. Testing standardize_chembl_bioactivity...")
    bioactivity_raw = {
        'activity_id': '12345',
        'assay_chembl_id': 'CHEMBL123',
        'target_chembl_id': 'CHEMBL2095173',
        'molecule_chembl_id': 'CHEMBL25',
        'standard_type': 'IC50',
        'standard_value': 1500,
        'standard_units': 'nM',
        'standard_relation': '=',
        'assay_type': 'B'
    }
    bioactivity = standardizer.standardize_chembl_bioactivity(bioactivity_raw)
    assert bioactivity.activity_type == 'IC50'
    assert bioactivity.activity_value == 1500.0
    assert bioactivity.activity_units == 'nM'
    print(f"   ✅ Bioactivity: {bioactivity.activity_type} = {bioactivity.activity_value} {bioactivity.activity_units}")

    # Test 3: standardize_chembl_target
    print("\n3. Testing standardize_chembl_target...")
    target_raw = {
        'target_chembl_id': 'CHEMBL2095173',
        'pref_name': 'AXL receptor tyrosine kinase',
        'target_type': 'SINGLE PROTEIN',
        'target_components': [{
            'component_symbol': 'AXL',
            'accession': 'P30530'
        }]
    }
    protein = standardizer.standardize_chembl_target(target_raw)
    assert protein.gene_symbol == 'AXL'
    assert protein.uniprot_id == 'P30530'
    print(f"   ✅ Target: {protein.gene_symbol} ({protein.uniprot_id})")

    # Test 4: convert_bioactivity_units
    print("\n4. Testing convert_bioactivity_units...")
    value_um = 1.5
    value_nm = standardizer.convert_bioactivity_units(value_um, 'uM', 'nM')
    assert value_nm == 1500.0
    print(f"   ✅ Unit conversion: {value_um} uM = {value_nm} nM")

    # Test 5: calculate_compound_confidence
    print("\n5. Testing calculate_compound_confidence...")
    confidence = standardizer.calculate_compound_confidence(compound_raw, 'chembl')
    assert 0.0 <= confidence <= 1.0
    print(f"   ✅ Compound confidence: {confidence:.2f}")

    # Test 6: aggregate_bioactivities
    print("\n6. Testing aggregate_bioactivities...")
    bioactivities = [bioactivity, bioactivity]  # Duplicate for testing
    aggregated = standardizer.aggregate_bioactivities(bioactivities)
    assert aggregated['median_ic50'] == 1500.0
    assert aggregated['activity_count'] == 2
    print(f"   ✅ Aggregated: {aggregated['activity_count']} activities, median IC50 = {aggregated['median_ic50']} nM")

    # Test 7: assess_drug_likeness_comprehensive
    print("\n7. Testing assess_drug_likeness_comprehensive...")
    drug_like = standardizer.assess_drug_likeness_comprehensive(compound)
    assert drug_like.lipinski_compliant  # Aspirin is drug-like
    assert drug_like.ro5_violations == 0
    print(f"   ✅ Drug-likeness: {drug_like.overall_assessment} (score: {drug_like.drug_likeness_score:.2f})")

    # Test 8: merge_kegg_chembl_drugs
    print("\n8. Testing merge_kegg_chembl_drugs...")
    kegg_drugs = [DrugInfo(
        drug_id='D00109',
        name='ASPIRIN',
        indication='Pain relief',
        mechanism='COX inhibitor',
        targets=['PTGS1', 'PTGS2'],
        development_status='approved',
        drug_class='NSAID',
        approval_status='approved'
    )]
    chembl_drugs = [
        compound,  # Should be deduplicated
        Compound(
            chembl_id='CHEMBL59',
            name='IBUPROFEN',
            molecular_weight=206.28,
            alogp=3.5,
            hba=2,
            hbd=1,
            psa=37.3,
            rtb=4,
            ro5_violations=0
        )
    ]
    merged = standardizer.merge_kegg_chembl_drugs(kegg_drugs, chembl_drugs)
    # Should have KEGG aspirin + ChEMBL ibuprofen (aspirin deduplicated)
    assert len(merged) == 2
    print(f"   ✅ Merged drugs: {len(kegg_drugs)} KEGG + {len(chembl_drugs)} ChEMBL = {len(merged)} total")

    print("\n" + "=" * 80)
    print("✅ ALL STANDARDIZATION TESTS PASSED!")
    print("=" * 80)
    print("\nPhase 2 standardization methods are working correctly!")

    return True

if __name__ == "__main__":
    success = test_standardizers()
    sys.exit(0 if success else 1)
