#!/usr/bin/env python3
"""Fix specific line indentation"""

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix lines 164-174
lines[163] = '            self.model = RandomForestClassifier(\n'  # Line 164 (0-indexed 163)
lines[171] = '            )\n'  # Line 172 (0-indexed 171)
lines[172] = '            self.model.fit(X_train_scaled, y_train)\n'  # Line 173 (0-indexed 172)
lines[173] = '            y_pred = self.model.predict(X_test_scaled)\n'  # Line 174 (0-indexed 173)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Fixed indentation for lines 164-174")

