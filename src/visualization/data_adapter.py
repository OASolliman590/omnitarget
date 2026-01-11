"""
Data Adapter for Visualization

Maps actual OmniTarget JSON structure to expected visualization format.
"""

import math
from typing import Dict, Any, List, Tuple


class VisualizationDataAdapter:
    """Adapts OmniTarget results to visualization-friendly format."""
    
    @staticmethod
    def adapt_scenario_1(data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt Scenario 1 data structure."""
        adapted = data.copy()
        
        # Map enrichment_results to pathway_enrichment
        # enrichment_results is a dict with nested categories, need to flatten
        if 'enrichment_results' in data and 'pathway_enrichment' not in data:
            enrichment_results = data['enrichment_results']
            
            # Extract pathways from nested enrichment structure
            pathway_list = []
            
            if isinstance(enrichment_results, dict):
                # Check if it has nested 'enrichment' dict
                if 'enrichment' in enrichment_results:
                    enrichment = enrichment_results['enrichment']
                    if isinstance(enrichment, dict):
                        # Flatten all categories (KEGG, Process, Component, etc.)
                        for category, terms in enrichment.items():
                            if isinstance(terms, list):
                                pathway_list.extend(terms)
                        
                        # Prioritize KEGG pathways if available
                        if 'KEGG' in enrichment and isinstance(enrichment['KEGG'], list):
                            # Put KEGG pathways first
                            kegg_pathways = enrichment['KEGG']
                            other_pathways = [p for p in pathway_list if p not in kegg_pathways]
                            pathway_list = kegg_pathways + other_pathways
                else:
                    # enrichment_results is a dict but not nested - check if it's already a list-like structure
                    # Try to extract values that look like pathway lists
                    for key, value in enrichment_results.items():
                        if isinstance(value, list) and value:
                            # Check if first item looks like a pathway dict
                            if isinstance(value[0], dict) and 'term' in value[0]:
                                pathway_list.extend(value)
            
            # If we couldn't extract pathways, try to use pathways directly
            if not pathway_list:
                pathways = data.get('pathways', [])
                if isinstance(pathways, list):
                    pathway_list = pathways
            
            adapted['pathway_enrichment'] = pathway_list
        
        # Ensure pathway_enrichment exists and is a list
        if 'pathway_enrichment' not in adapted:
            adapted['pathway_enrichment'] = data.get('pathways', [])
        elif not isinstance(adapted['pathway_enrichment'], list):
            # If it's still a dict, convert to empty list
            adapted['pathway_enrichment'] = []
        
        # Flatten enrichment categories for visualization
        enrichment_results = data.get('enrichment_results', {})
        flattened = []

        if isinstance(enrichment_results, dict):
            categories = enrichment_results.get('enrichment', {})

            if isinstance(categories, dict):
                for category, terms in categories.items():
                    if not isinstance(terms, list):
                        continue
                    for term in terms:
                        if not isinstance(term, dict):
                            continue
                        entry = term.copy()
                        entry['category'] = category
                        try:
                            p_val_raw = entry.get('p_value', entry.get('fdr', 1.0))
                            p_val = float(p_val_raw) if p_val_raw not in (None, '') else 1.0
                        except (ValueError, TypeError):
                            p_val = 1.0
                        entry['neg_log_p'] = -math.log10(p_val) if p_val > 0 else 10.0
                        flattened.append(entry)

        # Prioritize KEGG / WikiPathways entries, then others by significance
        prioritized = []
        for preferred in ('KEGG', 'WikiPathways'):
            preferred_terms = [entry for entry in flattened if entry.get('category') == preferred]
            prioritized.extend(preferred_terms)

        remaining = [entry for entry in flattened if entry not in prioritized]
        remaining.sort(key=lambda e: e.get('neg_log_p', 0.0), reverse=True)
        prioritized.extend(remaining)

        if prioritized:
            adapted['pathway_enrichment'] = prioritized[:30]
        elif 'pathways' in data:
            adapted['pathway_enrichment'] = data['pathways']

        adapted['network_nodes_count'] = len(data.get('network_nodes', []))
        adapted['network_edges_count'] = len(data.get('network_edges', []))

        return adapted
    
    @staticmethod
    def adapt_scenario_2(data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt Scenario 2 data structure."""
        adapted = data.copy()
        
        # Map pathways to pathway_memberships
        if 'pathways' in data and 'pathway_memberships' not in data:
            adapted['pathway_memberships'] = data['pathways']
        
        # Convert expression_profiles list to dict if needed
        if 'expression_profiles' in data:
            profiles = data['expression_profiles']
            if isinstance(profiles, list):
                # Convert list of ExpressionProfile objects to dict
                profile_dict = {}
                for prof in profiles:
                    if isinstance(prof, dict):
                        tissue = prof.get('tissue_type', prof.get('tissue', 'Unknown'))
                        level = prof.get('expression_level', prof.get('level', 'unknown'))
                        profile_dict[tissue] = level
                adapted['expression_profile'] = profile_dict
            elif isinstance(profiles, dict):
                adapted['expression_profile'] = profiles
        
        # Ensure gene field exists
        if 'target' in data and isinstance(data['target'], dict):
            adapted['gene'] = data['target'].get('gene_symbol', 'Unknown')
            adapted['target_gene'] = data['target'].get('gene_symbol', 'Unknown')
        
        return adapted
    
    @staticmethod
    def adapt_scenario_3(data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt Scenario 3 data structure."""
        adapted = data.copy()
        
        # prognostic_markers and prioritized_targets should already be correct
        # But ensure they exist
        if 'prognostic_markers' not in adapted:
            adapted['prognostic_markers'] = []
        
        if 'prioritized_targets' not in adapted:
            adapted['prioritized_targets'] = []
        
        # Ensure cancer_markers exists
        if 'cancer_markers' not in adapted:
            # Use prognostic_markers as fallback
            adapted['cancer_markers'] = adapted.get('prognostic_markers', [])
        
        return adapted
    
    @staticmethod
    def adapt_scenario_4(data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt Scenario 4 data structure with enhanced MRA data extraction."""
        adapted = data.copy()
        
        individual_results = data.get('individual_results', [])
        
        # Parse feedback loops from individual_results
        all_feedback_loops = []
        all_affected_nodes = {}
        network_edges = []
        
        for result in individual_results:
            if not isinstance(result, dict):
                continue
            
            target = result.get('target_node', result.get('target', 'Unknown'))
            mode = result.get('mode', 'inhibit')
            
            # Extract feedback loops - these come as strings like "AXL -> JAK2 -> GRB2"
            loops = result.get('feedback_loops', [])
            for loop in loops:
                if isinstance(loop, str):
                    # Parse "AXL -> JAK2 -> GRB2" format
                    nodes = [n.strip() for n in loop.split('->')]
                    all_feedback_loops.append({
                        'nodes': nodes,
                        'type': 'regulatory',
                        'source_target': target,
                        'original': loop
                    })
                    # Add edges from this loop
                    for i in range(len(nodes) - 1):
                        network_edges.append({
                            'source': nodes[i],
                            'target': nodes[i + 1],
                            'weight': 0.8,
                            'loop_origin': target
                        })
                elif isinstance(loop, dict):
                    all_feedback_loops.append(loop)
            
            # Aggregate affected_nodes across all targets
            affected = result.get('affected_nodes', {})
            if isinstance(affected, dict):
                for node, effect in affected.items():
                    if node not in all_affected_nodes:
                        all_affected_nodes[node] = {
                            'effect': 0.0,
                            'affected_by': [],
                            'max_effect': 0.0
                        }
                    effect_val = float(effect) if isinstance(effect, (int, float)) else 0.0
                    all_affected_nodes[node]['effect'] += effect_val
                    all_affected_nodes[node]['affected_by'].append(target)
                    all_affected_nodes[node]['max_effect'] = max(
                        all_affected_nodes[node]['max_effect'],
                        abs(effect_val)
                    )
            
            # Build network topology from direct_targets and downstream
            direct = result.get('direct_targets', [])
            for dt in direct:
                if isinstance(dt, str):
                    network_edges.append({
                        'source': target,
                        'target': dt,
                        'weight': 1.0,
                        'edge_type': 'direct'
                    })
        
        # Populate network_perturbation structure for existing visualizers
        adapted['network_perturbation'] = {
            'feedback_loops': all_feedback_loops,
            'affected_edges': network_edges,
            'loop_count': len(all_feedback_loops)
        }
        
        # Build combined_effects for visualization
        adapted['combined_effects'] = adapted.get('combined_effects', {})
        if not adapted['combined_effects'].get('affected_nodes'):
            adapted['combined_effects']['affected_nodes'] = all_affected_nodes
        
        # Extract centrality and network metrics for visualization
        centrality_data = []
        for result in individual_results:
            if not isinstance(result, dict):
                continue
            target = result.get('target_node', 'Unknown')
            network_impact = result.get('network_impact', {})
            centrality_data.append({
                'target': target,
                'network_centrality': network_impact.get('network_centrality', 0),
                'betweenness_centrality': network_impact.get('betweenness_centrality', 0),
                'total_affected': network_impact.get('total_affected', 0),
                'mean_effect': network_impact.get('mean_effect', 0),
                'propagation_depth': network_impact.get('propagation_depth', 0),
                'network_coverage': network_impact.get('network_coverage', 0)
            })
        adapted['centrality_data'] = centrality_data
        
        # Extract drug info for annotations
        drug_annotations = []
        for result in individual_results:
            if not isinstance(result, dict):
                continue
            drug_info = result.get('drug_info', {})
            if drug_info:
                drug_annotations.append({
                    'target': result.get('target_node', 'Unknown'),
                    'approved_drugs': drug_info.get('approved_drugs', []),
                    'clinical_trials': drug_info.get('clinical_trials', []),
                    'mechanism': drug_info.get('mechanism', 'Unknown'),
                    'indication': drug_info.get('indication', 'Unknown')
                })
        adapted['drug_annotations'] = drug_annotations
        
        return adapted

    
    @staticmethod
    def adapt_scenario_5(data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt Scenario 5 data structure."""
        adapted = data.copy()
        
        # Extract pathway information
        if 'kegg_pathways' in data or 'reactome_pathways' in data:
            kegg = data.get('kegg_pathways', [])
            reactome = data.get('reactome_pathways', [])
            
            # Set pathway_a and pathway_b
            if kegg and not adapted.get('pathway_a'):
                adapted['pathway_a'] = kegg[0] if isinstance(kegg, list) else kegg
            if reactome and not adapted.get('pathway_b'):
                adapted['pathway_b'] = reactome[0] if isinstance(reactome, list) else reactome
        
        # Extract gene overlap
        # gene_overlap may have counts (numbers) instead of lists - extract genes from pathways
        # First, extract all genes from pathways
        kegg_genes = []
        reactome_genes = []
        
        for pathway in kegg:
            if isinstance(pathway, dict):
                genes = pathway.get('genes', [])
                if isinstance(genes, list):
                    kegg_genes.extend(genes)
        
        for pathway in reactome:
            if isinstance(pathway, dict):
                genes = pathway.get('genes', [])
                if isinstance(genes, list):
                    reactome_genes.extend(genes)
        
        # Now check gene_overlap structure
        if 'gene_overlap' in data:
            overlap = data['gene_overlap']
            if isinstance(overlap, dict):
                # Check if it has gene lists or just counts
                common_genes_from_overlap = overlap.get('common_genes', [])
                
                # If common_genes is a number (count), use extracted genes
                if isinstance(common_genes_from_overlap, (int, float)):
                    # gene_overlap has counts, extract actual genes from pathways
                    common_genes = list(set(kegg_genes) & set(reactome_genes))
                    adapted['common_genes'] = common_genes
                    adapted['pathway_a_genes'] = list(set(kegg_genes) - set(common_genes))
                    adapted['pathway_b_genes'] = list(set(reactome_genes) - set(common_genes))
                else:
                    # gene_overlap has lists
                    adapted['common_genes'] = common_genes_from_overlap if isinstance(common_genes_from_overlap, list) else []
                    adapted['pathway_a_genes'] = overlap.get('kegg_only', overlap.get('kegg_unique', overlap.get('pathway_a_genes', [])))
                    adapted['pathway_b_genes'] = overlap.get('reactome_only', overlap.get('reactome_unique', overlap.get('pathway_b_genes', [])))
                
                # Extract metrics
                adapted['jaccard_similarity'] = overlap.get('jaccard_similarity', 0.0)
                adapted['overlap_coefficient'] = overlap.get('overlap_coefficient', overlap.get('overlap_percentage', 0.0))
            else:
                # If gene_overlap is not a dict, extract genes from pathways directly
                common_genes = list(set(kegg_genes) & set(reactome_genes))
                adapted['common_genes'] = common_genes
                adapted['pathway_a_genes'] = list(set(kegg_genes) - set(common_genes))
                adapted['pathway_b_genes'] = list(set(reactome_genes) - set(common_genes))
        else:
            # No gene_overlap - extract from pathways
            common_genes = list(set(kegg_genes) & set(reactome_genes))
            adapted['common_genes'] = common_genes
            adapted['pathway_a_genes'] = list(set(kegg_genes) - set(common_genes))
            adapted['pathway_b_genes'] = list(set(reactome_genes) - set(common_genes))
        
        # Calculate Jaccard similarity if not present
        if 'jaccard_similarity' not in adapted or not adapted.get('jaccard_similarity'):
            all_genes = set(kegg_genes) | set(reactome_genes)
            if all_genes:
                adapted['jaccard_similarity'] = len(adapted.get('common_genes', [])) / len(all_genes)
            else:
                adapted['jaccard_similarity'] = 0.0
        
        # Extract pathway overlap metrics
        if 'pathway_overlap' in data:
            overlap = data['pathway_overlap']
            if isinstance(overlap, dict):
                adapted['pathway_concordance'] = overlap.get('concordance', overlap.get('similarity', 0.0))
        
        # Normalize expression context for heatmaps
        expression_context = data.get('expression_context', {})
        raw_profiles = expression_context.get('expression_profiles', [])
        normalized_profiles = []

        for profile in raw_profiles[:30]:
            if hasattr(profile, 'model_dump'):
                normalized_profiles.append(profile.model_dump())
            elif isinstance(profile, dict):
                normalized_profiles.append(profile)

        if normalized_profiles:
            adapted['expression_heatmap'] = normalized_profiles
            adapted['expression_coverage'] = expression_context.get('coverage', 0.0)

        return adapted
    
    @staticmethod
    def adapt_scenario_6(data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt Scenario 6 data structure."""
        adapted = data.copy()
        
        # Ensure candidate_drugs exists
        if 'candidate_drugs' not in adapted:
            adapted['candidate_drugs'] = []
        
        # Map repurposing_scores if needed
        if 'repurposing_scores' in data and isinstance(data['repurposing_scores'], dict):
            # Already in correct format
            pass
        
        candidate_records = []
        for candidate in data.get('candidate_drugs', []):
            if hasattr(candidate, 'model_dump'):
                record = candidate.model_dump()
            elif isinstance(candidate, dict):
                record = candidate.copy()
            else:
                record = {
                    'drug_id': getattr(candidate, 'drug_id', 'unknown'),
                    'drug_name': getattr(candidate, 'drug_name', getattr(candidate, 'name', 'unknown')),
                    'repurposing_score': getattr(candidate, 'repurposing_score', 0.0),
                    'safety_profile': getattr(candidate, 'safety_profile', {}) or {},
                    'efficacy_prediction': getattr(candidate, 'efficacy_prediction', 0.0),
                    'target_protein': getattr(candidate, 'target_protein', None)
                }

            safety_profile = record.get('safety_profile', {}) or {}
            approval_status = str(safety_profile.get('approval_status', '')).lower()

            entry = {
                'drug_id': record.get('drug_id', record.get('id', 'unknown')),
                'drug_name': record.get('drug_name', record.get('name', 'unknown')),
                'repurposing_score': record.get('repurposing_score', record.get('efficacy_prediction', 0.0)),
                'efficacy_score': record.get('efficacy_prediction', record.get('repurposing_score', 0.0)),
                'safety_score': safety_profile.get('overall_score', safety_profile.get('score', 0.0)),
                'approval_status': approval_status,
                'target_protein': record.get('target_protein'),
                'targets': record.get('targets') or record.get('known_targets') or (
                    [record.get('target_protein')] if record.get('target_protein') else []
                )
            }
            candidate_records.append(entry)

        # Sort by repurposing score descending and trim for visualization
        candidate_records.sort(key=lambda c: c.get('repurposing_score', 0.0), reverse=True)
        adapted['candidate_drugs'] = candidate_records
        adapted['top_candidates'] = candidate_records[:50]

        off_target = data.get('off_target_analysis', {})
        entries = off_target.get('off_target_analysis') or off_target.get('entries') or []
        if entries:
            entries_sorted = sorted(
                entries,
                key=lambda e: (
                    e.get('off_target_potential', 0.0),
                    e.get('network_centrality', 0.0)
                ),
                reverse=True
            )
            adapted['high_risk_targets'] = entries_sorted[:15]
            adapted['off_target_coverage'] = off_target.get('coverage')

        net_validation = data.get('network_validation', {})
        if isinstance(net_validation, dict):
            if not net_validation.get('network_nodes'):
                net_validation['network_nodes'] = len(data.get('expression_validation', {}).get('profiles', []))
            adapted['network_validation'] = net_validation

        return adapted
    
    @classmethod
    def adapt(cls, scenario_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt data for specific scenario.
        
        Args:
            scenario_id: Scenario ID (1-6)
            data: Original scenario data
            
        Returns:
            Adapted data structure
        """
        adapters = {
            1: cls.adapt_scenario_1,
            2: cls.adapt_scenario_2,
            3: cls.adapt_scenario_3,
            4: cls.adapt_scenario_4,
            5: cls.adapt_scenario_5,
            6: cls.adapt_scenario_6,
        }
        
        adapter = adapters.get(scenario_id)
        if adapter:
            return adapter(data)
        
        return data.copy()
