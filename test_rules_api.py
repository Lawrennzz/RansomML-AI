#!/usr/bin/env python3
"""
Automated Test Script for Edit Rule and Delete Rule Functionality
Tests the rules API endpoints programmatically
"""

import requests
import json
import time

# Configuration
BASE_URL = 'http://localhost:5000'
TEST_USER = {
    'email': 'cyber_pro@example.com',
    'password': 'cyber123'
}

class RulesAPITester:
    def __init__(self):
        self.session = requests.Session()
        self.base_url = BASE_URL
        
    def login(self):
        """Login and establish session"""
        print("🔐 Logging in...")
        response = self.session.post(
            f'{self.base_url}/login',
            json=TEST_USER
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Login successful as {data['user']['role_name']}")
                return True
        print(f"❌ Login failed: {response.text}")
        return False
    
    def get_rules(self):
        """Get all rules"""
        print("\n📋 Fetching rules...")
        response = self.session.get(f'{self.base_url}/api/rules')
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                rules = data.get('rules', [])
                print(f"✅ Found {len(rules)} rules")
                return rules
        print(f"❌ Failed to get rules: {response.text}")
        return []
    
    def test_edit_rule_name(self):
        """Test Case 1: Edit rule name"""
        print("\n" + "="*60)
        print("TEST 1: Edit Rule - Change Name")
        print("="*60)
        
        rules = self.get_rules()
        if not rules:
            print("❌ No rules found to edit")
            return False
        
        # Get first rule
        original_rule = rules[0].copy()
        rule_id = original_rule['id']
        new_name = f"Updated Rule - {int(time.time())}"
        
        print(f"📝 Editing rule: {rule_id}")
        print(f"   Original name: {original_rule['name']}")
        print(f"   New name: {new_name}")
        
        # Edit rule
        rule_data = {
            'id': rule_id,
            'name': new_name,
            'when_prediction_is': original_rule.get('when_prediction_is', 'any'),
            'recommendation': original_rule.get('recommendation', 'MONITOR'),
            'enabled': original_rule.get('enabled', True),
            'conditions': original_rule.get('conditions', {})
        }
        
        response = self.session.post(
            f'{self.base_url}/api/rules',
            json=rule_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                # Verify update
                updated_rules = data.get('rules', [])
                updated_rule = next((r for r in updated_rules if r['id'] == rule_id), None)
                
                if updated_rule and updated_rule['name'] == new_name:
                    print(f"✅ Rule name updated successfully")
                    print(f"   Verified: {updated_rule['name']}")
                    return True
                else:
                    print(f"❌ Rule name not updated correctly")
                    return False
            else:
                print(f"❌ Update failed: {data.get('message')}")
                return False
        else:
            print(f"❌ Request failed: {response.status_code} - {response.text}")
            return False
    
    def test_edit_rule_conditions(self):
        """Test Case 2: Edit rule conditions"""
        print("\n" + "="*60)
        print("TEST 2: Edit Rule - Change Conditions")
        print("="*60)
        
        rules = self.get_rules()
        if not rules:
            print("❌ No rules found to edit")
            return False
        
        original_rule = rules[0].copy()
        rule_id = original_rule['id']
        
        # New conditions
        new_conditions = {
            'ExportSize': {'gt': 1000},
            'ResourceSize': {'gt': 1000}
        }
        
        print(f"📝 Editing rule: {rule_id}")
        print(f"   Original conditions: {json.dumps(original_rule.get('conditions', {}))}")
        print(f"   New conditions: {json.dumps(new_conditions)}")
        
        rule_data = {
            'id': rule_id,
            'name': original_rule.get('name', 'Test Rule'),
            'when_prediction_is': original_rule.get('when_prediction_is', 'any'),
            'recommendation': original_rule.get('recommendation', 'MONITOR'),
            'enabled': original_rule.get('enabled', True),
            'conditions': new_conditions
        }
        
        response = self.session.post(
            f'{self.base_url}/api/rules',
            json=rule_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                updated_rules = data.get('rules', [])
                updated_rule = next((r for r in updated_rules if r['id'] == rule_id), None)
                
                if updated_rule:
                    conditions_match = updated_rule['conditions'] == new_conditions
                    if conditions_match:
                        print(f"✅ Rule conditions updated successfully")
                        print(f"   Verified: {json.dumps(updated_rule['conditions'])}")
                        return True
                    else:
                        print(f"❌ Conditions don't match")
                        return False
                else:
                    print(f"❌ Rule not found after update")
                    return False
            else:
                print(f"❌ Update failed: {data.get('message')}")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False
    
    def test_edit_rule_recommendation(self):
        """Test Case 3: Edit rule recommendation"""
        print("\n" + "="*60)
        print("TEST 3: Edit Rule - Change Recommendation")
        print("="*60)
        
        rules = self.get_rules()
        if not rules:
            print("❌ No rules found to edit")
            return False
        
        original_rule = rules[0].copy()
        rule_id = original_rule['id']
        new_recommendation = 'NORMAL'
        
        print(f"📝 Editing rule: {rule_id}")
        print(f"   Original recommendation: {original_rule.get('recommendation')}")
        print(f"   New recommendation: {new_recommendation}")
        
        rule_data = {
            'id': rule_id,
            'name': original_rule.get('name', 'Test Rule'),
            'when_prediction_is': original_rule.get('when_prediction_is', 'any'),
            'recommendation': new_recommendation,
            'enabled': original_rule.get('enabled', True),
            'conditions': original_rule.get('conditions', {})
        }
        
        response = self.session.post(
            f'{self.base_url}/api/rules',
            json=rule_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                updated_rules = data.get('rules', [])
                updated_rule = next((r for r in updated_rules if r['id'] == rule_id), None)
                
                if updated_rule and updated_rule['recommendation'] == new_recommendation:
                    print(f"✅ Rule recommendation updated successfully")
                    return True
                else:
                    print(f"❌ Recommendation not updated correctly")
                    return False
            else:
                print(f"❌ Update failed: {data.get('message')}")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False
    
    def test_disable_rule(self):
        """Test Case 4: Disable rule"""
        print("\n" + "="*60)
        print("TEST 4: Edit Rule - Disable Rule")
        print("="*60)
        
        rules = self.get_rules()
        if not rules:
            print("❌ No rules found to edit")
            return False
        
        original_rule = rules[0].copy()
        rule_id = original_rule['id']
        
        print(f"📝 Disabling rule: {rule_id}")
        
        rule_data = {
            'id': rule_id,
            'name': original_rule.get('name', 'Test Rule'),
            'when_prediction_is': original_rule.get('when_prediction_is', 'any'),
            'recommendation': original_rule.get('recommendation', 'MONITOR'),
            'enabled': False,  # Disable
            'conditions': original_rule.get('conditions', {})
        }
        
        response = self.session.post(
            f'{self.base_url}/api/rules',
            json=rule_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                updated_rules = data.get('rules', [])
                updated_rule = next((r for r in updated_rules if r['id'] == rule_id), None)
                
                if updated_rule and updated_rule['enabled'] == False:
                    print(f"✅ Rule disabled successfully")
                    return True
                else:
                    print(f"❌ Rule not disabled correctly")
                    return False
            else:
                print(f"❌ Update failed: {data.get('message')}")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False
    
    def test_create_new_rule(self):
        """Test Case 9: Create new rule"""
        print("\n" + "="*60)
        print("TEST 5: Create New Rule")
        print("="*60)
        
        rules_before = self.get_rules()
        count_before = len(rules_before)
        
        new_rule = {
            'name': f'Test Rule - {int(time.time())}',
            'when_prediction_is': 'any',
            'recommendation': 'MONITOR',
            'enabled': True,
            'conditions': {
                'Machine': {'eq': 332}
            }
        }
        
        print(f"📝 Creating new rule: {new_rule['name']}")
        
        response = self.session.post(
            f'{self.base_url}/api/rules',
            json=new_rule,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                rules_after = data.get('rules', [])
                count_after = len(rules_after)
                
                if count_after > count_before:
                    new_rule_id = data.get('updated_id')
                    created_rule = next((r for r in rules_after if r['id'] == new_rule_id), None)
                    
                    if created_rule:
                        print(f"✅ New rule created successfully")
                        print(f"   Rule ID: {new_rule_id}")
                        print(f"   Rule name: {created_rule['name']}")
                        return True, new_rule_id
                    else:
                        print(f"❌ Rule created but not found in list")
                        return False, None
                else:
                    print(f"❌ Rule count didn't increase")
                    return False, None
            else:
                print(f"❌ Creation failed: {data.get('message')}")
                return False, None
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False, None
    
    def test_delete_rule(self, rule_id=None):
        """Test Case 5: Delete rule"""
        print("\n" + "="*60)
        print("TEST 6: Delete Rule")
        print("="*60)
        
        rules_before = self.get_rules()
        count_before = len(rules_before)
        
        if not rule_id:
            if not rules_before:
                print("❌ No rules found to delete")
                return False
            rule_id = rules_before[0]['id']
        
        print(f"🗑️  Deleting rule: {rule_id}")
        
        response = self.session.delete(
            f'{self.base_url}/api/rules/{rule_id}'
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                rules_after = data.get('rules', [])
                count_after = len(rules_after)
                
                if count_after < count_before:
                    # Verify rule is gone
                    rule_exists = any(r['id'] == rule_id for r in rules_after)
                    if not rule_exists:
                        print(f"✅ Rule deleted successfully")
                        print(f"   Rules before: {count_before}, after: {count_after}")
                        return True
                    else:
                        print(f"❌ Rule still exists after deletion")
                        return False
                else:
                    print(f"❌ Rule count didn't decrease")
                    return False
            else:
                print(f"❌ Deletion failed: {data.get('message')}")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False
    
    def test_delete_nonexistent_rule(self):
        """Test Case 10: Delete non-existent rule"""
        print("\n" + "="*60)
        print("TEST 7: Delete Non-Existent Rule")
        print("="*60)
        
        fake_id = 'non-existent-rule-12345'
        print(f"🗑️  Attempting to delete: {fake_id}")
        
        response = self.session.delete(
            f'{self.base_url}/api/rules/{fake_id}'
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ API handled non-existent rule gracefully (idempotent)")
                return True
            else:
                print(f"❌ API returned error: {data.get('message')}")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False
    
    def run_all_tests(self):
        """Run all test cases"""
        print("="*60)
        print("RULES API TEST SUITE")
        print("="*60)
        
        if not self.login():
            print("\n❌ Cannot proceed without login")
            return
        
        results = []
        
        # Test 1: Edit rule name
        results.append(("Edit Rule Name", self.test_edit_rule_name()))
        
        # Test 2: Edit rule conditions
        results.append(("Edit Rule Conditions", self.test_edit_rule_conditions()))
        
        # Test 3: Edit rule recommendation
        results.append(("Edit Rule Recommendation", self.test_edit_rule_recommendation()))
        
        # Test 4: Disable rule
        results.append(("Disable Rule", self.test_disable_rule()))
        
        # Test 5: Create new rule
        success, new_rule_id = self.test_create_new_rule()
        results.append(("Create New Rule", success))
        
        # Test 6: Delete rule (use newly created rule)
        if new_rule_id:
            results.append(("Delete Rule", self.test_delete_rule(new_rule_id)))
        else:
            results.append(("Delete Rule", self.test_delete_rule()))
        
        # Test 7: Delete non-existent rule
        results.append(("Delete Non-Existent Rule", self.test_delete_nonexistent_rule()))
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {test_name}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        print("="*60)
        
        return passed == total

if __name__ == '__main__':
    tester = RulesAPITester()
    success = tester.run_all_tests()
    exit(0 if success else 1)

