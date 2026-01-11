#!/usr/bin/env python3
"""
Test workaround for ChEMBL - get drugs via activities
"""
import asyncio
import json
from src.core.mcp_client_manager import MCPClientManager

async def test_chembl_workaround():
    """Test getting drugs via activities instead of direct search"""

    print("Testing ChEMBL drug discovery via activities...")
    print("=" * 80)

    try:
        manager = MCPClientManager('config/mcp_servers.json')

        async with manager.session() as session:
            # Strategy: Get activities for a gene, then extract unique compounds
            gene = 'EGFR'

            # Get target ID first
            result = await session.chembl.search_targets(gene, limit=1)
            targets = result.get('targets', [])

            if not targets:
                print(f"❌ No targets found for {gene}")
                return

            target_id = targets[0].get('target_chembl_id')
            print(f"✅ Target ID: {target_id}")

            # Get activities
            result = await session.chembl.search_activities(
                target_chembl_id=target_id,
                limit=50
            )
            activities = result.get('activities', [])
            print(f"✅ Found {len(activities)} activities")

            # Extract unique compounds
            unique_compounds = {}
            for activity in activities:
                mol_id = activity.get('molecule_chembl_id')
                if mol_id and mol_id not in unique_compounds:
                    unique_compounds[mol_id] = {
                        'chembl_id': mol_id,
                        'name': activity.get('molecule_pref_name', 'Unknown'),
                        'smiles': activity.get('canonical_smiles', ''),
                        'ic50': activity.get('standard_value', ''),
                        'units': activity.get('standard_units', '')
                    }

            print(f"✅ Extracted {len(unique_compounds)} unique compounds")
            print(f"\nTop 5 compounds:")
            for i, (mol_id, data) in enumerate(list(unique_compounds.items())[:5]):
                print(f"  {i+1}. {data['name']} ({mol_id}) - IC50: {data['ic50']} {data['units']}")

            # Save to file
            with open(f'chembl_compounds_{gene.lower()}.json', 'w') as f:
                json.dump(list(unique_compounds.values()), f, indent=2)
            print(f"\n💾 Saved to chembl_compounds_{gene.lower()}.json")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_chembl_workaround())
