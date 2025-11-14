# Test Cases: Edit Rule and Delete Rule Functionality

## Overview
This document provides comprehensive test cases for editing and deleting detection rules in the Ransomware Detection System.

## Prerequisites
- Flask application running (`python app.py`)
- User logged in with `configure_detection_rules` permission:
  - **Cybersecurity Professional**: `cyber_pro@example.com` / `cyber123`
  - **IT Administrator**: `admin@example.com` / `admin123`

---

## Test Case 1: Edit Existing Rule - Change Rule Name

### Objective
Verify that an existing rule can be edited to change its name.

### Pre-Conditions
- User is logged in with appropriate permissions
- At least one rule exists in the system (default rules: `rule-1` or `rule-2`)

### Test Steps
1. Navigate to **"Rules"** section in the navbar
2. Click **"Refresh"** button to load existing rules
3. Locate a rule in the table (e.g., "High crypto ops => Immediate")
4. Click **"Edit"** button for that rule
5. In the prompt dialog:
   - **Rule name**: Change to "Updated: High crypto operations trigger"
   - **When prediction is**: Keep as "ransomware"
   - **Recommendation**: Keep as "IMMEDIATE_ACTION"
   - **Enable rule**: Click OK (Yes)
   - **Conditions**: Keep as `{"BitcoinAddresses":{"gt":0}}`
6. Click OK on all prompts
7. Verify the rule table refreshes automatically
8. Confirm the updated rule name appears in the table

### Expected Results
- ✅ Rule name is successfully updated
- ✅ Rule table refreshes and shows new name
- ✅ Rule ID remains unchanged
- ✅ Other rule properties remain unchanged
- ✅ Success message or no error displayed

### API Test (Alternative Method)
```python
import requests
import json

# Login
session = requests.Session()
login_data = {
    'email': 'cyber_pro@example.com',
    'password': 'cyber123'
}
session.post('http://localhost:5000/login', json=login_data)

# Edit rule
rule_data = {
    'id': 'rule-1',
    'name': 'Updated: High crypto operations trigger',
    'when_prediction_is': 'ransomware',
    'recommendation': 'IMMEDIATE_ACTION',
    'enabled': True,
    'conditions': {
        'BitcoinAddresses': {'gt': 0}
    }
}

response = session.post(
    'http://localhost:5000/api/rules',
    json=rule_data,
    headers={'Content-Type': 'application/json'}
)

result = response.json()
print("Edit Result:", result)
assert result['success'] == True
assert any(r['name'] == 'Updated: High crypto operations trigger' for r in result['rules'])
```

---

## Test Case 2: Edit Rule - Change Conditions

### Objective
Verify that rule conditions can be modified.

### Pre-Conditions
- User is logged in with appropriate permissions
- Rule `rule-2` exists (High file modifications => Monitor)

### Test Steps
1. Navigate to **"Rules"** section
2. Click **"Refresh"** to load rules
3. Click **"Edit"** on rule `rule-2`
4. In the prompts:
   - **Rule name**: Keep as "High file modifications => Monitor"
   - **When prediction is**: Keep as "ransomware"
   - **Recommendation**: Keep as "MONITOR"
   - **Enable rule**: Yes
   - **Conditions**: Change to `{"ExportSize":{"gt":1000},"ResourceSize":{"gt":1000}}`
5. Complete all prompts
6. Verify rule updates in table

### Expected Results
- ✅ Conditions are updated successfully
- ✅ Rule table shows updated conditions
- ✅ Rule remains enabled
- ✅ JSON validation works correctly

### API Test
```python
# Edit rule conditions
rule_data = {
    'id': 'rule-2',
    'name': 'High file modifications => Monitor',
    'when_prediction_is': 'ransomware',
    'recommendation': 'MONITOR',
    'enabled': True,
    'conditions': {
        'ExportSize': {'gt': 1000},
        'ResourceSize': {'gt': 1000}
    }
}

response = session.post('http://localhost:5000/api/rules', json=rule_data)
result = response.json()
print("Updated Conditions:", result['rules'][1]['conditions'])
assert result['rules'][1]['conditions']['ExportSize']['gt'] == 1000
```

---

## Test Case 3: Edit Rule - Change Recommendation

### Objective
Verify that rule recommendation can be changed.

### Test Steps
1. Navigate to **"Rules"** section
2. Click **"Edit"** on any rule
3. Change **Recommendation** to "NORMAL"
4. Complete all prompts
5. Verify the change

### Expected Results
- ✅ Recommendation is updated
- ✅ Rule table reflects new recommendation
- ✅ Rule evaluation uses new recommendation

### API Test
```python
rule_data = {
    'id': 'rule-1',
    'name': 'High crypto ops => Immediate',
    'when_prediction_is': 'ransomware',
    'recommendation': 'NORMAL',  # Changed from IMMEDIATE_ACTION
    'enabled': True,
    'conditions': {'BitcoinAddresses': {'gt': 0}}
}

response = session.post('http://localhost:5000/api/rules', json=rule_data)
result = response.json()
assert result['rules'][0]['recommendation'] == 'NORMAL'
```

---

## Test Case 4: Edit Rule - Disable Rule

### Objective
Verify that a rule can be disabled without deleting it.

### Test Steps
1. Navigate to **"Rules"** section
2. Click **"Edit"** on an enabled rule
3. When asked **"Enable this rule?"**, click **Cancel** (No)
4. Complete other prompts
5. Verify rule shows as disabled in table

### Expected Results
- ✅ Rule `enabled` field is set to `False`
- ✅ Rule table shows "No" in Enabled column
- ✅ Disabled rule is not evaluated during detection
- ✅ Rule still exists and can be re-enabled

### API Test
```python
rule_data = {
    'id': 'rule-1',
    'name': 'High crypto ops => Immediate',
    'when_prediction_is': 'ransomware',
    'recommendation': 'IMMEDIATE_ACTION',
    'enabled': False,  # Disable rule
    'conditions': {'BitcoinAddresses': {'gt': 0}}
}

response = session.post('http://localhost:5000/api/rules', json=rule_data)
result = response.json()
assert result['rules'][0]['enabled'] == False
```

---

## Test Case 5: Delete Rule - Confirm Deletion

### Objective
Verify that a rule can be deleted with confirmation.

### Pre-Conditions
- At least one rule exists

### Test Steps
1. Navigate to **"Rules"** section
2. Click **"Refresh"** to load rules
3. Note the number of rules
4. Click **"Delete"** button on a rule
5. In confirmation dialog, click **OK** (Confirm)
6. Verify rule is removed from table
7. Verify rule count decreased by 1

### Expected Results
- ✅ Confirmation dialog appears
- ✅ Rule is deleted after confirmation
- ✅ Rule table refreshes automatically
- ✅ Deleted rule no longer appears
- ✅ Other rules remain intact

### API Test
```python
# Get current rules
response = session.get('http://localhost:5000/api/rules')
rules_before = response.json()['rules']
initial_count = len(rules_before)

# Delete rule
rule_id = 'rule-1'
response = session.delete(f'http://localhost:5000/api/rules/{rule_id}')
result = response.json()

# Verify deletion
assert result['success'] == True
assert len(result['rules']) == initial_count - 1
assert not any(r['id'] == rule_id for r in result['rules'])
```

---

## Test Case 6: Delete Rule - Cancel Deletion

### Objective
Verify that deletion can be cancelled.

### Test Steps
1. Navigate to **"Rules"** section
2. Click **"Delete"** on a rule
3. In confirmation dialog, click **Cancel**
4. Verify rule still exists in table

### Expected Results
- ✅ Rule is NOT deleted
- ✅ Rule remains in table
- ✅ No changes to rules list

---

## Test Case 7: Edit Rule - Invalid JSON Conditions

### Objective
Verify error handling for invalid JSON in conditions.

### Test Steps
1. Navigate to **"Rules"** section
2. Click **"Edit"** on a rule
3. In **Conditions** prompt, enter invalid JSON: `{"ExportSize":{gt:500}}` (missing quotes)
4. Complete prompts

### Expected Results
- ✅ Error alert: "Invalid conditions JSON"
- ✅ Rule is NOT updated
- ✅ Original rule remains unchanged

---

## Test Case 8: Edit Rule - Cancel During Edit

### Objective
Verify that cancelling any prompt during edit cancels the operation.

### Test Steps
1. Navigate to **"Rules"** section
2. Click **"Edit"** on a rule
3. In **Rule name** prompt, click **Cancel**
4. Verify no changes are made

### Expected Results
- ✅ Edit operation is cancelled
- ✅ No changes to rule
- ✅ Rule table unchanged

---

## Test Case 9: Create New Rule via Edit

### Objective
Verify that clicking "New Rule" creates a new rule.

### Test Steps
1. Navigate to **"Rules"** section
2. Click **"New Rule"** button
3. Fill in all prompts:
   - **Name**: "Test New Rule"
   - **When prediction is**: "any"
   - **Recommendation**: "MONITOR"
   - **Enable**: Yes
   - **Conditions**: `{"Machine":{"eq":332}}`
4. Complete all prompts
5. Verify new rule appears in table

### Expected Results
- ✅ New rule is created with unique ID
- ✅ Rule appears in table
- ✅ Rule is enabled
- ✅ Rule can be edited/deleted

### API Test
```python
# Create new rule
new_rule = {
    'name': 'Test New Rule',
    'when_prediction_is': 'any',
    'recommendation': 'MONITOR',
    'enabled': True,
    'conditions': {'Machine': {'eq': 332}}
}

response = session.post('http://localhost:5000/api/rules', json=new_rule)
result = response.json()
assert result['success'] == True
assert result['updated_id'] is not None
assert any(r['name'] == 'Test New Rule' for r in result['rules'])
```

---

## Test Case 10: Delete Non-Existent Rule

### Objective
Verify error handling when deleting a rule that doesn't exist.

### Test Steps (API)
```python
# Try to delete non-existent rule
response = session.delete('http://localhost:5000/api/rules/non-existent-rule')
result = response.json()
# Should succeed but return empty or unchanged rules list
assert result['success'] == True
```

### Expected Results
- ✅ API returns success (idempotent operation)
- ✅ Rules list unchanged
- ✅ No error thrown

---

## Test Case 11: Edit Rule - Change When Prediction Is

### Objective
Verify that "when_prediction_is" field can be changed.

### Test Steps
1. Edit a rule that triggers on "ransomware"
2. Change **When prediction is** to "benign"
3. Complete prompts
4. Verify change

### Expected Results
- ✅ Field is updated
- ✅ Rule now triggers on benign predictions
- ✅ Rule evaluation logic uses new value

### API Test
```python
rule_data = {
    'id': 'rule-1',
    'name': 'Test Rule',
    'when_prediction_is': 'benign',  # Changed
    'recommendation': 'MONITOR',
    'enabled': True,
    'conditions': {}
}

response = session.post('http://localhost:5000/api/rules', json=rule_data)
result = response.json()
assert result['rules'][0]['when_prediction_is'] == 'benign'
```

---

## Test Case 12: Multiple Rule Edits in Sequence

### Objective
Verify that multiple rules can be edited in sequence.

### Test Steps
1. Edit rule-1, change name to "Rule 1 Updated"
2. Edit rule-2, change name to "Rule 2 Updated"
3. Verify both changes persist

### Expected Results
- ✅ Both rules are updated
- ✅ Changes persist after refresh
- ✅ No conflicts or data loss

---

## Test Case 13: Permission Check - Edit Rule Without Permission

### Objective
Verify that users without permission cannot edit rules.

### Pre-Conditions
- Login as System User (no `configure_detection_rules` permission)

### Test Steps
1. Login as `user@example.com` / `user123`
2. Try to access Rules section
3. Verify access is denied

### Expected Results
- ✅ Rules section is hidden or shows 403 error
- ✅ API returns 403 Forbidden
- ✅ Error message indicates permission denied

### API Test
```python
# Login as user without permission
session = requests.Session()
login_data = {'email': 'user@example.com', 'password': 'user123'}
session.post('http://localhost:5000/login', json=login_data)

# Try to edit rule
response = session.post('http://localhost:5000/api/rules', json={...})
assert response.status_code == 403
```

---

## Test Case 14: Rules Persistence After Server Restart

### Objective
Verify that edited/deleted rules persist after server restart.

### Test Steps
1. Edit a rule
2. Delete a rule
3. Restart Flask server
4. Verify changes persist

### Expected Results
- ✅ Edited rules remain edited
- ✅ Deleted rules remain deleted
- ✅ Rules are loaded from `rules.json` file

---

## Test Case 15: Complex Conditions - Multiple Features

### Objective
Verify that rules with complex conditions work correctly.

### Test Steps
1. Create/edit rule with multiple conditions:
   ```json
   {
     "ExportSize": {"gt": 500},
     "ResourceSize": {"gt": 500},
     "BitcoinAddresses": {"eq": 0},
     "NumberOfSections": {"gte": 5}
   }
   ```
2. Verify rule saves correctly
3. Test rule evaluation

### Expected Results
- ✅ Complex conditions are saved correctly
- ✅ Rule evaluates all conditions
- ✅ JSON is properly formatted

---

## Summary

### Test Coverage
- ✅ Edit rule name
- ✅ Edit rule conditions
- ✅ Edit rule recommendation
- ✅ Enable/disable rule
- ✅ Delete rule with confirmation
- ✅ Cancel deletion
- ✅ Create new rule
- ✅ Error handling (invalid JSON, missing permissions)
- ✅ Rule persistence
- ✅ Complex conditions

### Test Methods
1. **UI Testing**: Manual testing via web interface
2. **API Testing**: Automated testing via Python requests
3. **Integration Testing**: Verify rules affect detection behavior

### Files to Test
- `app.py`: `/api/rules` endpoints (GET, POST, DELETE)
- `templates/index.html`: Rules UI (editRule, deleteRule functions)
- `rules.json`: Rule persistence file

