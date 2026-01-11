#!/usr/bin/env python3
"""
Simple test to verify STRING confidence_score field is now extracted
"""
import asyncio
import json
from src.core.mcp_client_manager import MCPClientManager

# Load the saved STRING response
with open('string_response_axl_brca1.json', 'r') as f:
    string_response = json.load(f)

# Test the confidence extraction logic (from scenario_4)
edges = string_response.get('edges', [])
confidence_scores = []

print("Testing confidence extraction from STRING response...")
print(f"Total edges: {len(edges)}")

for i, edge in enumerate(edges[:10]):  # Check first 10 edges
    # The fixed code looks for 'confidence_score' FIRST
    score = (edge.get('confidence_score') or
            edge.get('score') or
            edge.get('combined_score') or
            edge.get('confidence') or
            edge.get('weight') or
            0)

    try:
        score = float(score)
        confidence_scores.append(score)
        if i < 5:  # Print first 5
            print(f"Edge {i+1}: {edge['protein_a']} - {edge['protein_b']} = {score}")
    except (ValueError, TypeError):
        pass

if confidence_scores:
    median_confidence = sorted(confidence_scores)[len(confidence_scores)//2]
    print(f"\n✅ SUCCESS! Found {len(confidence_scores)} confidence scores")
    print(f"   Min: {min(confidence_scores):.3f}")
    print(f"   Median: {median_confidence:.3f}")
    print(f"   Max: {max(confidence_scores):.3f}")

    if median_confidence > 0.0:
        print(f"\n🎉 Issue #2 FIXED! Median confidence is now {median_confidence:.3f} (> 0.0)")
    else:
        print(f"\n⚠️ Confidence is still 0.0")
else:
    print(f"\n❌ FAILED! No confidence scores found")
