"""
Unit Tests for UniProt Domain Parsing

Tests for Node.js UniProt MCP domain feature parsing and extraction.
"""

import pytest

pytestmark = pytest.mark.unit


class TestUniProtDomainParsing:
    """Test UniProt domain feature parsing."""
    
    def test_unwrap_node_response_format(self):
        """Test unwrapping Node.js MCP response format."""
        from src.mcp_clients.uniprot_client import UniProtClient
        
        # Mock Node.js response format
        node_response = {
            'content': [{
                'type': 'text',
                'text': '{"primaryAccession": "P04637", "features": [{"type": "Domain", "description": "DNA binding"}]}'
            }]
        }
        
        client = UniProtClient("/dummy/path.js")
        unwrapped = client._unwrap_node_response(node_response)
        
        assert isinstance(unwrapped, dict)
        assert unwrapped.get('primaryAccession') == 'P04637'
        assert 'features' in unwrapped
    
    def test_unwrap_python_response_format(self):
        """Test that Python MCP responses pass through unchanged."""
        from src.mcp_clients.uniprot_client import UniProtClient
        
        # Python response format (direct dict)
        python_response = {
            'accession': 'P04637',
            'features': [{'type': 'Domain', 'description': 'DNA binding'}]
        }
        
        client = UniProtClient("/dummy/path")  # Not .js, so Python format
        unwrapped = client._unwrap_node_response(python_response)
        
        assert unwrapped == python_response
    
    def test_extract_primary_accession(self):
        """Test extracting primaryAccession from search_by_gene response."""
        from src.mcp_clients.uniprot_client import UniProtClient
        
        # Node.js search_by_gene response format
        search_response = {
            'content': [{
                'type': 'text',
                'text': '{"results": [{"primaryAccession": "P04637", "uniProtkbId": "P53_HUMAN"}]}'
            }]
        }
        
        client = UniProtClient("/dummy/path.js")
        unwrapped = client._unwrap_node_response(search_response)
        
        assert unwrapped.get('results')
        assert unwrapped['results'][0]['primaryAccession'] == 'P04637'
    
    def test_parse_domain_features_structure(self):
        """Test parsing domain features structure from get_protein_features."""
        # Mock domain features response
        features_response = {
            'features': [
                {'type': 'Domain', 'description': 'DNA binding', 'location': {'start': {'value': 102}, 'end': {'value': 292}}},
                {'type': 'Binding site', 'description': 'ATP binding', 'location': {'start': {'value': 1}, 'end': {'value': 100}}}
            ],
            'domains': [],
            'activeSites': [],
            'bindingSites': [
                {'type': 'Binding site', 'description': 'Protein binding'}
            ]
        }
        
        # Extract features
        all_features = []
        all_features.extend(features_response.get('features', []))
        all_features.extend(features_response.get('domains', []))
        all_features.extend(features_response.get('activeSites', []))
        all_features.extend(features_response.get('bindingSites', []))
        
        assert len(all_features) == 3
        assert any('DNA binding' in str(f.get('description', '')) for f in all_features)
        assert any('ATP binding' in str(f.get('description', '')) for f in all_features)
    
    def test_druggable_keyword_matching(self):
        """Test druggable keyword matching logic."""
        druggable_keywords = {
            'kinase': 0.15,
            'atp': 0.12,
            'catalytic': 0.10,
            'binding': 0.08,
            'receptor': 0.08,
            'enzyme': 0.06,
            'active': 0.06,
            'domain': 0.04,
        }
        
        # Test features
        features = [
            {'type': 'Domain', 'description': 'Protein kinase domain'},
            {'type': 'Binding site', 'description': 'ATP binding site'},
            {'type': 'Region', 'description': 'Catalytic domain'},
            {'type': 'Domain', 'description': 'DNA binding domain'}
        ]
        
        domain_boost = 0.0
        for feature in features:
            feature_type = feature.get('type', '').lower()
            description = feature.get('description', '').lower()
            
            for keyword, boost_value in druggable_keywords.items():
                if keyword in feature_type or keyword in description:
                    domain_boost = max(domain_boost, boost_value)
                    break
        
        assert domain_boost == 0.15  # Highest boost from kinase
        assert domain_boost > 0.12  # Should be higher than ATP
    
    def test_location_extraction(self):
        """Test extracting start/end locations from domain features."""
        feature = {
            'type': 'Domain',
            'description': 'DNA binding',
            'location': {
                'start': {'value': 102},
                'end': {'value': 292}
            }
        }
        
        location = feature.get('location', {})
        start_val = None
        end_val = None
        
        if isinstance(location, dict):
            start_obj = location.get('start', {})
            end_obj = location.get('end', {})
            
            if isinstance(start_obj, dict):
                start_val = start_obj.get('value')
            if isinstance(end_obj, dict):
                end_val = end_obj.get('value')
        
        assert start_val == 102
        assert end_val == 292
    
    def test_domain_boost_scoring(self):
        """Test domain boost scoring calculation."""
        # Simulate domain boost logic
        domain_boost = 0.0
        
        features = [
            {'type': 'Binding site', 'description': 'ATP binding'},
            {'type': 'Domain', 'description': 'protein kinase'}
        ]
        
        druggable_keywords = {
            'kinase': 0.15,
            'atp': 0.12,
            'binding': 0.08,
        }
        
        for feature in features:
            feature_type = feature.get('type', '').lower()
            description = feature.get('description', '').lower()
            
            for keyword, boost_value in druggable_keywords.items():
                if keyword in feature_type or keyword in description:
                    domain_boost = max(domain_boost, boost_value)
                    break
        
        assert domain_boost == 0.15  # kinase is highest
        
        # Test binding site boost
        binding_features = [
            {'type': 'Binding site', 'description': 'protein binding'}
        ]
        
        domain_boost_binding = 0.0
        for feature in binding_features:
            feature_type = feature.get('type', '').lower()
            description = feature.get('description', '').lower()
            
            for keyword, boost_value in druggable_keywords.items():
                if keyword in feature_type or keyword in description:
                    domain_boost_binding = max(domain_boost_binding, boost_value)
                    break
        
        assert domain_boost_binding == 0.08  # binding keyword


class TestUniProtIDResolution:
    """Test UniProt ID resolution from search_by_gene."""
    
    def test_extract_primary_accession_from_search(self):
        """Test extracting primaryAccession from search_by_gene results."""
        # Mock search_by_gene response (Node.js format)
        search_result = {
            'results': [
                {
                    'primaryAccession': 'P04637',
                    'uniProtkbId': 'P53_HUMAN',
                    'entryType': 'UniProtKB reviewed (Swiss-Prot)'
                }
            ]
        }
        
        if search_result.get('results') and len(search_result['results']) > 0:
            first_result = search_result['results'][0]
            uniprot_accession = first_result.get('primaryAccession') or first_result.get('accession') or first_result.get('id')
            
            assert uniprot_accession == 'P04637'
    
    def test_handle_empty_search_results(self):
        """Test handling empty search results."""
        search_result = {'results': []}
        
        uniprot_accession = None
        if search_result.get('results') and len(search_result['results']) > 0:
            first_result = search_result['results'][0]
            uniprot_accession = first_result.get('primaryAccession')
        
        assert uniprot_accession is None
    
    def test_fallback_accession_fields(self):
        """Test fallback to accession or id fields."""
        # Test with accession field
        result1 = {'accession': 'P04637'}
        accession1 = result1.get('primaryAccession') or result1.get('accession') or result1.get('id')
        assert accession1 == 'P04637'
        
        # Test with id field
        result2 = {'id': 'P04637'}
        accession2 = result2.get('primaryAccession') or result2.get('accession') or result2.get('id')
        assert accession2 == 'P04637'
        
        # Test with primaryAccession (preferred)
        result3 = {'primaryAccession': 'P04637', 'accession': 'P04638'}
        accession3 = result3.get('primaryAccession') or result3.get('accession') or result3.get('id')
        assert accession3 == 'P04637'  # Should prefer primaryAccession

