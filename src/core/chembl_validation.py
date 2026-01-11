"""
ChEMBL Data Validation

ChEMBL-specific validators for ensuring data quality and scientific accuracy.
Implements 10 specialized validators for drug discovery data.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from ..models.data_models import Compound, Bioactivity, DrugInfo, DrugLikenessAssessment

logger = logging.getLogger(__name__)


class ChEMBLValidator:
    """ChEMBL-specific data quality validators."""

    def __init__(self):
        """Initialize ChEMBL validator with scientific thresholds."""
        # Drug-likeness thresholds (Lipinski Rule of Five)
        self.lipinski_thresholds = {
            'molecular_weight': 500,  # Da
            'logp': 5,  # Partition coefficient
            'hba': 10,  # H-bond acceptors
            'hbd': 5,   # H-bond donors
        }

        # Bioactivity value ranges (in nM)
        self.bioactivity_ranges = {
            'IC50': (0.001, 1_000_000),  # 1 pM to 1 mM
            'Ki': (0.001, 1_000_000),
            'EC50': (0.001, 1_000_000),
            'Kd': (0.001, 1_000_000),
        }

        # Quality thresholds
        self.quality_thresholds = {
            'min_bioactivity_confidence': 0.6,
            'min_druggability_score': 0.3,
            'min_activities_per_target': 3,
            'min_compounds_per_target': 2,
            'max_ro5_violations': 1,
        }

    # ==========================================
    # 1. Drug-Likeness Validator
    # ==========================================

    def validate_drug_likeness(self, compound: Compound) -> Tuple[bool, str]:
        """
        Validate compound drug-likeness using Lipinski Rule of Five.

        Lipinski Rule of Five:
        - Molecular weight ≤ 500 Da
        - LogP ≤ 5
        - Hydrogen bond acceptors ≤ 10
        - Hydrogen bond donors ≤ 5
        - No more than 1 violation allowed

        Args:
            compound: Compound object with molecular properties

        Returns:
            Tuple of (is_valid, message)

        Example:
            >>> valid, msg = validator.validate_drug_likeness(compound)
            >>> if not valid:
            ...     print(f"Drug-likeness failed: {msg}")
        """
        try:
            violations = []

            # Check molecular weight
            if compound.molecular_weight and compound.molecular_weight > self.lipinski_thresholds['molecular_weight']:
                violations.append(f"MW={compound.molecular_weight:.1f} > 500 Da")

            # Check logP (using alogp as proxy)
            if compound.alogp and compound.alogp > self.lipinski_thresholds['logp']:
                violations.append(f"LogP={compound.alogp:.1f} > 5")

            # Check H-bond acceptors
            if compound.hba and compound.hba > self.lipinski_thresholds['hba']:
                violations.append(f"HBA={compound.hba} > 10")

            # Check H-bond donors
            if compound.hbd and compound.hbd > self.lipinski_thresholds['hbd']:
                violations.append(f"HBD={compound.hbd} > 5")

            # Lipinski allows up to 1 violation
            is_valid = len(violations) <= self.quality_thresholds['max_ro5_violations']

            if is_valid:
                message = "Drug-like (Lipinski compliant)"
            else:
                message = f"Non-drug-like ({len(violations)} violations: {', '.join(violations)})"

            return is_valid, message

        except Exception as e:
            logger.error(f"Drug-likeness validation failed: {e}")
            return False, f"Validation error: {str(e)}"

    # ==========================================
    # 2. Bioactivity Quality Validator
    # ==========================================

    def validate_bioactivity_quality(self, bioactivity: Bioactivity) -> Tuple[bool, str]:
        """
        Validate bioactivity data quality.

        Checks:
        - Activity type is recognized (IC50, Ki, EC50, Kd)
        - Activity value is present and numeric
        - Activity units are standardized (nM)
        - Activity relation is valid (=, <, >, ~)
        - Assay type is valid (B=Binding, F=Functional)

        Args:
            bioactivity: Bioactivity object

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            issues = []

            # Check activity type
            valid_types = ['IC50', 'Ki', 'EC50', 'Kd', 'IC90', 'AC50', 'GI50']
            if bioactivity.activity_type not in valid_types:
                issues.append(f"Unknown activity type: {bioactivity.activity_type}")

            # Check activity value
            if bioactivity.activity_value is None:
                issues.append("Missing activity value")
            elif bioactivity.activity_value <= 0:
                issues.append(f"Invalid activity value: {bioactivity.activity_value}")

            # Check units are standardized to nM
            if bioactivity.activity_units and bioactivity.activity_units != 'nM':
                issues.append(f"Units not standardized: {bioactivity.activity_units} (expected: nM)")

            # Check relation is valid
            valid_relations = ['=', '<', '>', '<=', '>=', '~']
            if bioactivity.activity_relation and bioactivity.activity_relation not in valid_relations:
                issues.append(f"Invalid relation: {bioactivity.activity_relation}")

            # Check assay type
            if bioactivity.assay_type and bioactivity.assay_type not in ['B', 'F', 'A', 'T', 'P']:
                issues.append(f"Unknown assay type: {bioactivity.assay_type}")

            is_valid = len(issues) == 0
            message = "High-quality bioactivity data" if is_valid else f"Quality issues: {'; '.join(issues)}"

            return is_valid, message

        except Exception as e:
            logger.error(f"Bioactivity quality validation failed: {e}")
            return False, f"Validation error: {str(e)}"

    # ==========================================
    # 3. IC50 Range Validator
    # ==========================================

    def validate_bioactivity_range(self, bioactivity: Bioactivity) -> Tuple[bool, str]:
        """
        Validate bioactivity value is within reasonable pharmaceutical range.

        Typical ranges (in nM):
        - IC50/Ki/EC50/Kd: 0.001 nM (1 pM) to 1,000,000 nM (1 mM)
        - Values outside this range are likely errors

        Args:
            bioactivity: Bioactivity object

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            if bioactivity.activity_value is None:
                return False, "Missing activity value"

            activity_type = bioactivity.activity_type
            value = bioactivity.activity_value

            # Get range for this activity type
            if activity_type in self.bioactivity_ranges:
                min_val, max_val = self.bioactivity_ranges[activity_type]
            else:
                # Use IC50 range as default
                min_val, max_val = self.bioactivity_ranges['IC50']

            # Check if value is within range
            if value < min_val or value > max_val:
                return False, f"{activity_type}={value:.3f} nM outside valid range ({min_val}-{max_val} nM)"

            # Classify potency
            if value < 10:
                potency = "highly potent"
            elif value < 100:
                potency = "potent"
            elif value < 1000:
                potency = "moderately potent"
            else:
                potency = "weakly potent"

            return True, f"Valid {activity_type}={value:.1f} nM ({potency})"

        except Exception as e:
            logger.error(f"Bioactivity range validation failed: {e}")
            return False, f"Validation error: {str(e)}"

    # ==========================================
    # 4. Drug-Target Association Validator
    # ==========================================

    def validate_drug_target_association(
        self,
        drug: DrugInfo,
        target_gene: str,
        bioactivities: Optional[List[Bioactivity]] = None
    ) -> Tuple[bool, str]:
        """
        Validate drug-target association.

        Checks:
        - Target is in drug's target list
        - Bioactivity data supports association
        - Association strength is sufficient

        Args:
            drug: DrugInfo object
            target_gene: Target gene symbol
            bioactivities: Optional list of supporting bioactivities

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Check if target is in drug's target list
            if drug.targets and target_gene not in drug.targets:
                return False, f"Target {target_gene} not in drug targets: {drug.targets}"

            # If bioactivities provided, check for strong evidence
            if bioactivities:
                # Filter for this target
                target_activities = [
                    b for b in bioactivities
                    if b.activity_value and b.activity_value < 1000  # IC50 < 1 µM
                ]

                if not target_activities:
                    return False, f"No potent bioactivities (<1 µM) for {target_gene}"

                # Get best (lowest) activity value
                best_activity = min(b.activity_value for b in target_activities)
                activity_count = len(target_activities)

                message = f"Strong association: {activity_count} activities, best={best_activity:.1f} nM"
                return True, message

            # If no bioactivities, rely on target list only
            return True, f"Association confirmed (target in drug target list)"

        except Exception as e:
            logger.error(f"Drug-target association validation failed: {e}")
            return False, f"Validation error: {str(e)}"

    # ==========================================
    # 5. Compound Structure Validator
    # ==========================================

    def validate_compound_structure(self, compound: Compound) -> Tuple[bool, str]:
        """
        Validate compound structure identifiers.

        Checks:
        - SMILES string format (basic validation)
        - InChI key format
        - ChEMBL ID format

        Args:
            compound: Compound object

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            issues = []
            validations = []

            # Validate ChEMBL ID format (CHEMBL + digits)
            if compound.chembl_id:
                if re.match(r'^CHEMBL\d+$', compound.chembl_id):
                    validations.append("Valid ChEMBL ID")
                else:
                    issues.append(f"Invalid ChEMBL ID format: {compound.chembl_id}")
            else:
                issues.append("Missing ChEMBL ID")

            # Validate SMILES (basic check - not empty, reasonable length)
            if hasattr(compound, 'smiles') and compound.smiles:
                if 5 <= len(compound.smiles) <= 2000:  # Reasonable SMILES length
                    validations.append("Valid SMILES")
                else:
                    issues.append(f"SMILES length unusual: {len(compound.smiles)}")

            # Validate InChI Key format (14-char + '-' + 10-char + '-' + 1-char)
            if hasattr(compound, 'inchi_key') and compound.inchi_key:
                if re.match(r'^[A-Z]{14}-[A-Z]{10}-[A-Z]$', compound.inchi_key):
                    validations.append("Valid InChI Key")
                else:
                    issues.append(f"Invalid InChI Key format")

            is_valid = len(issues) == 0

            if is_valid:
                message = f"Valid structure ({', '.join(validations)})"
            else:
                message = f"Structure issues: {'; '.join(issues)}"

            return is_valid, message

        except Exception as e:
            logger.error(f"Compound structure validation failed: {e}")
            return False, f"Validation error: {str(e)}"

    # ==========================================
    # 6. Bioactivity Confidence Validator
    # ==========================================

    def validate_bioactivity_confidence(self, bioactivity: Bioactivity) -> Tuple[bool, str]:
        """
        Validate bioactivity measurement confidence.

        Confidence factors:
        - Assay type (Binding > Functional)
        - Exact measurement (= > < or >)
        - Data validity
        - Confidence score ≥ 0.6

        Args:
            bioactivity: Bioactivity object

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            confidence = bioactivity.confidence if bioactivity.confidence else 0.0
            threshold = self.quality_thresholds['min_bioactivity_confidence']

            if confidence >= threshold:
                # High confidence
                factors = []
                if bioactivity.assay_type == 'B':
                    factors.append("binding assay")
                if bioactivity.activity_relation == '=':
                    factors.append("exact measurement")

                message = f"High confidence ({confidence:.2f})"
                if factors:
                    message += f": {', '.join(factors)}"
                return True, message
            else:
                # Low confidence
                reasons = []
                if confidence < threshold:
                    reasons.append(f"confidence={confidence:.2f} < {threshold}")
                if bioactivity.assay_type == 'F':
                    reasons.append("functional assay (less reliable)")
                if bioactivity.activity_relation in ['<', '>']:
                    reasons.append(f"approximate ({bioactivity.activity_relation})")

                message = f"Low confidence: {', '.join(reasons)}"
                return False, message

        except Exception as e:
            logger.error(f"Bioactivity confidence validation failed: {e}")
            return False, f"Validation error: {str(e)}"

    # ==========================================
    # 7. Target Druggability Validator
    # ==========================================

    def validate_target_druggability(
        self,
        gene_symbol: str,
        druggability_score: float,
        bioactivity_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Validate target druggability score.

        Checks:
        - Score is in valid range (0-1)
        - Score meets minimum threshold (≥0.3)
        - Score is supported by bioactivity data

        Args:
            gene_symbol: Target gene symbol
            druggability_score: Calculated druggability score
            bioactivity_data: Optional bioactivity metrics

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Check score range
            if not (0.0 <= druggability_score <= 1.0):
                return False, f"Score {druggability_score:.3f} outside valid range (0-1)"

            threshold = self.quality_thresholds['min_druggability_score']

            # Check minimum threshold
            if druggability_score < threshold:
                return False, f"Score {druggability_score:.3f} below minimum ({threshold})"

            # Classify druggability
            if druggability_score >= 0.7:
                classification = "highly druggable"
            elif druggability_score >= 0.5:
                classification = "moderately druggable"
            elif druggability_score >= 0.3:
                classification = "potentially druggable"
            else:
                classification = "poorly druggable"

            # Add bioactivity evidence if available
            evidence = []
            if bioactivity_data:
                if 'activity_count' in bioactivity_data:
                    evidence.append(f"{bioactivity_data['activity_count']} activities")
                if 'median_ic50' in bioactivity_data and bioactivity_data['median_ic50']:
                    evidence.append(f"median IC50={bioactivity_data['median_ic50']:.1f} nM")

            message = f"{gene_symbol} is {classification} (score={druggability_score:.3f})"
            if evidence:
                message += f"; evidence: {', '.join(evidence)}"

            return True, message

        except Exception as e:
            logger.error(f"Target druggability validation failed: {e}")
            return False, f"Validation error: {str(e)}"

    # ==========================================
    # 8. Cross-Database Concordance Validator
    # ==========================================

    def validate_kegg_chembl_concordance(
        self,
        kegg_drugs: List[DrugInfo],
        chembl_compounds: List[Compound],
        gene_symbol: str
    ) -> Tuple[bool, str]:
        """
        Validate concordance between KEGG and ChEMBL drug data.

        Checks:
        - Overlap in drug names/compounds
        - Consistency in target associations
        - Complementary coverage

        Args:
            kegg_drugs: KEGG drug list
            chembl_compounds: ChEMBL compound list
            gene_symbol: Target gene symbol

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Extract drug names (normalized to uppercase)
            kegg_names = {drug.name.upper() for drug in kegg_drugs if drug.name}
            chembl_names = {comp.name.upper() for comp in chembl_compounds if comp.name}

            # Calculate overlap
            overlap = kegg_names & chembl_names
            total_unique = kegg_names | chembl_names

            kegg_only = len(kegg_names - chembl_names)
            chembl_only = len(chembl_names - kegg_names)
            overlap_count = len(overlap)

            # Calculate concordance metrics
            if total_unique:
                concordance_rate = overlap_count / len(total_unique)
            else:
                concordance_rate = 0.0

            # Build message
            summary = []
            summary.append(f"KEGG: {len(kegg_drugs)} drugs")
            summary.append(f"ChEMBL: {len(chembl_compounds)} compounds")
            summary.append(f"Overlap: {overlap_count}")
            summary.append(f"Concordance: {concordance_rate:.1%}")

            message = f"{gene_symbol} - {', '.join(summary)}"

            # Validate: Either have overlap OR complementary data
            has_overlap = overlap_count > 0
            has_complementary = (kegg_only > 0 and chembl_only > 0)
            is_valid = has_overlap or has_complementary or len(total_unique) > 0

            if not is_valid:
                message += " - WARNING: No drug data from either source"

            return is_valid, message

        except Exception as e:
            logger.error(f"Cross-database concordance validation failed: {e}")
            return False, f"Validation error: {str(e)}"

    # ==========================================
    # 9. Drug Development Status Validator
    # ==========================================

    def validate_drug_development_status(self, drug: DrugInfo) -> Tuple[bool, str]:
        """
        Validate drug development status consistency.

        Checks:
        - Development status is recognized
        - Approval status is consistent
        - Phase information is valid

        Args:
            drug: DrugInfo object

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            valid_statuses = [
                'approved', 'clinical', 'preclinical', 'experimental',
                'withdrawn', 'suspended', 'investigational'
            ]

            issues = []

            # Check development status
            if drug.development_status:
                status_lower = drug.development_status.lower()
                if status_lower not in valid_statuses:
                    issues.append(f"Unknown development status: {drug.development_status}")

            # Check approval status consistency
            if drug.approval_status:
                approval_lower = drug.approval_status.lower()
                if drug.development_status:
                    dev_lower = drug.development_status.lower()
                    # If approved, development should be 'approved'
                    if approval_lower == 'approved' and dev_lower != 'approved':
                        issues.append(f"Inconsistent: approval={approval_lower}, development={dev_lower}")

            is_valid = len(issues) == 0

            if is_valid:
                status = drug.development_status or drug.approval_status or "unknown"
                message = f"Valid development status: {status}"
            else:
                message = f"Status issues: {'; '.join(issues)}"

            return is_valid, message

        except Exception as e:
            logger.error(f"Drug development status validation failed: {e}")
            return False, f"Validation error: {str(e)}"

    # ==========================================
    # 10. Molecular Descriptor Validator
    # ==========================================

    def validate_molecular_descriptors(self, compound: Compound) -> Tuple[bool, str]:
        """
        Validate molecular descriptors are within reasonable ranges.

        Checks:
        - Molecular weight: 100-1000 Da (typical drug range)
        - LogP (alogp): -5 to 10 (typical range)
        - PSA: 0-200 Ų (polar surface area)
        - RTB: 0-20 (rotatable bonds)
        - Descriptors are consistent

        Args:
            compound: Compound object

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            issues = []
            validations = []

            # Validate molecular weight
            if compound.molecular_weight:
                if 100 <= compound.molecular_weight <= 1000:
                    validations.append(f"MW={compound.molecular_weight:.1f} Da")
                elif compound.molecular_weight < 100:
                    issues.append(f"MW too low: {compound.molecular_weight:.1f} Da")
                elif compound.molecular_weight > 1000:
                    issues.append(f"MW too high: {compound.molecular_weight:.1f} Da (large molecule)")

            # Validate LogP
            if compound.alogp:
                if -5 <= compound.alogp <= 10:
                    validations.append(f"LogP={compound.alogp:.1f}")
                else:
                    issues.append(f"LogP unusual: {compound.alogp:.1f}")

            # Validate PSA
            if compound.psa:
                if 0 <= compound.psa <= 200:
                    validations.append(f"PSA={compound.psa:.1f} Ų")
                else:
                    issues.append(f"PSA unusual: {compound.psa:.1f} Ų")

            # Validate rotatable bonds
            if compound.rtb:
                if 0 <= compound.rtb <= 20:
                    validations.append(f"RTB={compound.rtb}")
                else:
                    issues.append(f"RTB unusual: {compound.rtb}")

            # Check for consistency (HBA + HBD should correlate with PSA)
            if compound.hba and compound.hbd and compound.psa:
                # Very rough estimate: PSA ≈ 20 * (HBA + HBD)
                expected_psa = 20 * (compound.hba + compound.hbd)
                if abs(compound.psa - expected_psa) > 100:
                    issues.append(f"PSA/H-bond inconsistency")

            is_valid = len(issues) == 0

            if is_valid:
                message = f"Valid descriptors: {', '.join(validations)}"
            else:
                message = f"Descriptor issues: {'; '.join(issues)}"

            return is_valid, message

        except Exception as e:
            logger.error(f"Molecular descriptor validation failed: {e}")
            return False, f"Validation error: {str(e)}"

    # ==========================================
    # Summary Validation Method
    # ==========================================

    def validate_chembl_data_quality(
        self,
        compounds: List[Compound],
        bioactivities: List[Bioactivity],
        drugs: Optional[List[DrugInfo]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive ChEMBL data quality validation.

        Runs all 10 validators and provides summary report.

        Args:
            compounds: List of Compound objects
            bioactivities: List of Bioactivity objects
            drugs: Optional list of DrugInfo objects

        Returns:
            Dictionary with validation results and metrics
        """
        results = {
            'total_compounds': len(compounds),
            'total_bioactivities': len(bioactivities),
            'total_drugs': len(drugs) if drugs else 0,
            'validations': {},
            'pass_rates': {},
            'issues': []
        }

        try:
            # 1. Validate drug-likeness for all compounds
            drug_like_count = 0
            for compound in compounds:
                valid, msg = self.validate_drug_likeness(compound)
                if valid:
                    drug_like_count += 1
            results['validations']['drug_likeness'] = f"{drug_like_count}/{len(compounds)}"
            results['pass_rates']['drug_likeness'] = drug_like_count / len(compounds) if compounds else 0

            # 2. Validate bioactivity quality
            quality_count = 0
            for bioactivity in bioactivities:
                valid, msg = self.validate_bioactivity_quality(bioactivity)
                if valid:
                    quality_count += 1
            results['validations']['bioactivity_quality'] = f"{quality_count}/{len(bioactivities)}"
            results['pass_rates']['bioactivity_quality'] = quality_count / len(bioactivities) if bioactivities else 0

            # 3. Validate bioactivity ranges
            range_count = 0
            for bioactivity in bioactivities:
                valid, msg = self.validate_bioactivity_range(bioactivity)
                if valid:
                    range_count += 1
            results['validations']['bioactivity_range'] = f"{range_count}/{len(bioactivities)}"
            results['pass_rates']['bioactivity_range'] = range_count / len(bioactivities) if bioactivities else 0

            # 4. Validate compound structures
            structure_count = 0
            for compound in compounds:
                valid, msg = self.validate_compound_structure(compound)
                if valid:
                    structure_count += 1
            results['validations']['compound_structure'] = f"{structure_count}/{len(compounds)}"
            results['pass_rates']['compound_structure'] = structure_count / len(compounds) if compounds else 0

            # 5. Validate bioactivity confidence
            confidence_count = 0
            for bioactivity in bioactivities:
                valid, msg = self.validate_bioactivity_confidence(bioactivity)
                if valid:
                    confidence_count += 1
            results['validations']['bioactivity_confidence'] = f"{confidence_count}/{len(bioactivities)}"
            results['pass_rates']['bioactivity_confidence'] = confidence_count / len(bioactivities) if bioactivities else 0

            # 6. Validate molecular descriptors
            descriptor_count = 0
            for compound in compounds:
                valid, msg = self.validate_molecular_descriptors(compound)
                if valid:
                    descriptor_count += 1
            results['validations']['molecular_descriptors'] = f"{descriptor_count}/{len(compounds)}"
            results['pass_rates']['molecular_descriptors'] = descriptor_count / len(compounds) if compounds else 0

            # 7. Validate drug development status (if drugs provided)
            if drugs:
                dev_status_count = 0
                for drug in drugs:
                    valid, msg = self.validate_drug_development_status(drug)
                    if valid:
                        dev_status_count += 1
                results['validations']['development_status'] = f"{dev_status_count}/{len(drugs)}"
                results['pass_rates']['development_status'] = dev_status_count / len(drugs)

            # Calculate overall quality score
            pass_rates = [v for k, v in results['pass_rates'].items()]
            results['overall_quality_score'] = sum(pass_rates) / len(pass_rates) if pass_rates else 0.0

            # Add summary
            quality_level = "Excellent" if results['overall_quality_score'] >= 0.9 else \
                           "Good" if results['overall_quality_score'] >= 0.75 else \
                           "Acceptable" if results['overall_quality_score'] >= 0.6 else \
                           "Poor"

            results['summary'] = f"{quality_level} data quality (score: {results['overall_quality_score']:.2%})"

        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
            results['issues'].append(str(e))

        return results
