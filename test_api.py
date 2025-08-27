#!/usr/bin/env python3
"""
Test script for the News Classification API
"""

import requests
import json
import sys

def test_api(api_url, headline):
    """Test the news classification API with a headline"""
    
    payload = {
        "query": {
            "headline": headline
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"🧪 Testing headline: '{headline}'")
        print(f"📡 Sending request to: {api_url}")
        
        response = requests.post(api_url, json=payload, headers=headers)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"🏷️  Predicted Label: {result.get('predicted_label', 'Unknown')}")
            print(f"📈 Probabilities: {result.get('probabilities', [])}")
            
            # Pretty print probabilities
            if 'probabilities' in result and result['probabilities']:
                probs = result['probabilities'][0]
                categories = ["Business", "Science", "Entertainment", "Health"]
                print("\n📊 Detailed Probabilities:")
                for i, (cat, prob) in enumerate(zip(categories, probs)):
                    print(f"   {cat}: {prob:.2%}")
        else:
            print("❌ Error!")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <API_URL> [headline]")
        print("\nExample:")
        print("python test_api.py https://xyz.execute-api.us-east-1.amazonaws.com/prod/classify")
        print('python test_api.py https://xyz.execute-api.us-east-1.amazonaws.com/prod/classify "Stock market crashes"')
        sys.exit(1)
    
    api_url = sys.argv[1]
    
    if len(sys.argv) >= 3:
        # Use provided headline
        headline = sys.argv[2]
        test_api(api_url, headline)
    else:
        # Test with multiple sample headlines
        test_headlines = [
            "Scientists discover new planet in distant galaxy",
            "Stock market reaches all-time high amid economic growth",
            "Hollywood star announces retirement from acting career", 
            "New breakthrough in cancer treatment shows promising results"
        ]
        
        print("🧪 Testing API with sample headlines...\n")
        for i, headline in enumerate(test_headlines, 1):
            print(f"--- Test {i}/4 ---")
            test_api(api_url, headline)
            print()

if __name__ == "__main__":
    main()
