# W4d: robot

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### actuation_data.tcpspeedval  (column)

- Current `meta.kb_ids`: `[23, 4, 14]`
- Bucket: **C** — R-SPLIT-TRINITY (helper + calc + classification, 3-way split)
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 23 [value_illustration] "TCP Speed"
    def: TCP speed is measured in mm/s. Higher speeds may be required for certain applications but can affect precision.
  - KB 4 [calculation_knowledge] "Average TCP Speed (ATCS)"
    def: For a given robot R, ATCS = \frac{\sum_{ad \in \text{actuation_data} \mid \text{actdetref = R}} tcpspeedval}{|\{ad \in \text{actuation_data} \mid \text{actdetref = R}\}|}
  - KB 14 [domain_knowledge] "Fast Robot"
    def: The robot is fast if ATCS > 1000.

### actuation_data.poserrmmval  (column)

- Current `meta.kb_ids`: `[22, 3, 13]`
- Bucket: **C** — R-SPLIT-TRINITY (helper + calc + classification, 3-way split)
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 22 [value_illustration] "Position Error"
    def: Position error is measured in millimeters. Lower values indicate higher precision.
  - KB 3 [calculation_knowledge] "Average Position Error (APE)"
    def: For a given robot R, APE = \frac{\sum_{ad \in \text{actuation_data} \mid \text{actdetref = R}} poserrmmval}{|\{ad \in \text{actuation_data} \mid \text{actdetref = R}\}|}
  - KB 13 [domain_knowledge] "Precision Category"
    def: If APE < 0.1, 'High Precision'; else if APE < 0.5, 'Medium Precision'; else 'Low Precision'.

### actuation_data.row_payload_ratio  (column)

- Current `meta.kb_ids`: `[39, 49]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 39 [calculation_knowledge] "Payload Utilization Ratio (PUR)"
    def: For a given robot R, PUR = \frac{\sum_{ad \in \text{actuation_data} \mid \text{actdetref = R}} payloadwval}{\text{(robot_details.payloadcapkg where botdetref = R)} \cdot \text{|}\{ad \in \text{actuation_data} \mid \text{actdetref = R}\}|}, \text{where RAY adjusts for age-related capacity changes if …
  - KB 49 [domain_knowledge] "Overloaded Robot"
    def: A robot R is Overloaded if PUR > 0.9 and RAY > 1.

### joint_condition.j1tempval  (column)

- Current `meta.kb_ids`: `[20, 1, 11]`
- Bucket: **C** — R-SPLIT-TRINITY (helper + calc + classification, 3-way split)
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 20 [value_illustration] "Joint Temperature"
    def: Joint temperatures are measured in degrees Celsius. Typical operating temperatures range from 20°C to 60°C. Temperatures above 60°C may indicate potential issues.
  - KB 1 [calculation_knowledge] "Average Joint 1 Temperature (AJ1T)"
    def: For a given robot R, AJ1T = \frac{\sum_{jc \in \text{joint_condition} \mid \text{jcdetref = R}} j1tempval}{|\{jc \in \text{joint_condition} \mid \text{jcdetref = R}\}|}
  - KB 11 [domain_knowledge] "High Temperature Joint 1"
    def: Joint 1 has high temperature if AJ1T > 50.

### joint_condition.row_max_joint_temp  (column)

- Current `meta.kb_ids`: `[2, 12]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 2 [calculation_knowledge] "Maximum Joint Temperature (MJT)"
    def: For a given robot R, MJT = \max_{jc \in \text{joint_condition} \mid \text{jcdetref = R}} \max(jc.j1tempval, jc.j2tempval, jc.j3tempval, jc.j4tempval, jc.j5tempval, jc.j6tempval)
  - KB 12 [domain_knowledge] "Overheating Risk"
    def: There is an overheating risk if MJT > 70.

### joint_performance.j1_torque  (column)

- Current `meta.kb_ids`: `[38, 48]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 38 [calculation_knowledge] "Joint Torque Variance (JTV)"
    def: For a given robot R, JTV = \frac{\sum_{jp \in \text{joint_performance} \mid \text{jperfdetref = R}} \sum_{i=1}^6 ((joint_metrics->>'jointi'->>'torque'::float - \mu_i)^2)}{\text{|}\{jp \in \text{joint_performance} \mid \text{jperfdetref = R}\}| \cdot 6}, \text{where } \mu_i = \frac{\sum_{jp} joint_me…
  - KB 48 [domain_knowledge] "Operational Instability"
    def: A robot R has Operational Instability if JTV > 50 and MJT > 60.

### maintenance_and_fault.rulhours  (column)

- Current `meta.kb_ids`: `[25, 6, 16]`
- Bucket: **C** — R-SPLIT-TRINITY (helper + calc + classification, 3-way split)
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 25 [value_illustration] "Remaining Useful Life (RUL)"
    def: RUL is measured in hours and indicates the estimated time until the next maintenance is required.
  - KB 6 [calculation_knowledge] "Minimum Remaining Useful Life (MRUL)"
    def: For a given robot R, MRUL = \min_{mf \in \text{maintenance_and_fault} \mid \text{upkeeprobot = R}} rulhours
  - KB 16 [domain_knowledge] "Urgent Maintenance Needed"
    def: Urgent maintenance is needed if MRUL < 100.

### maintenance_and_fault.weighted_faultpredscore  (column)

- Current `meta.kb_ids`: `[30, 40]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 30 [calculation_knowledge] "Weighted Fault Prediction Score (WFPS)"
    def: For a given robot R, WFPS = \frac{\sum_{mf \in \text{maintenance_and_fault} \mid \text{upkeeprobot = R}} (faultpredscore \cdot w(mf))}{\sum_{mf \in \text{maintenance_and_fault} \mid \text{upkeeprobot = R}} w(mf)}, \text{where } w(mf) = 1 / (1 + \text{upkeepduedays})
  - KB 40 [domain_knowledge] "Maintenance Priority Level"
    def: For a robot R, the Maintenance Priority Level is: - 'CRITICAL' if WFPS > 0.6 AND MRUL < 500 - 'WARNING' if WFPS > 0.4 OR MRUL < 500 - 'NORMAL' otherwise

### maintenance_and_fault.weighted_upkeepcostest  (column)

- Current `meta.kb_ids`: `[35, 45]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 35 [calculation_knowledge] "Maintenance Cost Trend (MCT)"
    def: For a given robot R, MCT = \frac{\sum_{mf \in \text{maintenance_and_fault} \mid \text{upkeeprobot = R}} \text{upkeepcostest} \cdot w(mf)}{\text{|}\{mf \in \text{maintenance_and_fault} \mid \text{upkeeprobot = R}\}|}, \text{where } w(mf) = \frac{1}{1 + \text{upkeepduedays}}
  - KB 45 [domain_knowledge] "Escalating Maintenance Costs"
    def: A robot R has Escalating Maintenance Costs if MCT > 500 and RAY > 2.

### operation.operreg  (column)

- Current `meta.kb_ids`: `[8, 18]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 8 [calculation_knowledge] "Number of Operations (NO)"
    def: For a given robot R, NO = |\{o \in \text{operation} \mid \text{operbotdetref = R}\}|
  - KB 18 [domain_knowledge] "Multi-Operation Robot"
    def: The robot has performed multiple operations if NO > 1.

### operation.totopshrval  (column)

- Current `meta.kb_ids`: `[7, 17]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 7 [calculation_knowledge] "Total Operating Hours (TOH)"
    def: For a given robot R, TOH = \max_{o \in \text{operation} \mid \text{operbotdetref = R}} totopshrval
  - KB 17 [domain_knowledge] "Heavily Used Robot"
    def: The robot is heavily used if TOH > 10000.

### operation.progcyclecount  (column)

- Current `meta.kb_ids`: `[9, 19]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 9 [calculation_knowledge] "Total Program Cycles (TPC)"
    def: For a given robot R, TPC = \sum_{o \in \text{operation} \mid \text{operbotdetref = R}} progcyclecount
  - KB 19 [domain_knowledge] "High Cycle Count Robot"
    def: The robot has a high cycle count if TPC > 1000000.

### operation.cycletimesecval  (column)

- Current `meta.kb_ids`: `[33, 43, 51]`
- Bucket: **D** — R-SPLIT-MULTI-FORMULA (one entity per formula)
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 33 [calculation_knowledge] "Operation Cycle Efficiency (OCE)"
    def: For a given robot R, OCE = \frac{\text{TPC}}{\sum_{o \in \text{operation} \mid \text{operbotdetref = R}} cycletimesecval}
  - KB 43 [domain_knowledge] "Cycle Efficiency Category"
    def: For a robot R, the Cycle Efficiency Category is: - 'Low Efficiency' if OCE < 100 AND TPC > 500000 - 'Medium Efficiency' if OCE < 150 OR TPC > 300000 - 'High Efficiency' otherwise
  - KB 51 [calculation_knowledge] "Average Cycle Time"
    def: For a given robot R, Average Cycle Time = \frac{\sum_{o \in \text{operation} \mid \text{operbotdetref = R}} \text{cycletimesecval}}{|\{o \in \text{operation} \mid \text{operbotdetref = R}\}|}

### performance_and_safety.row_safety_incident_total  (column)

- Current `meta.kb_ids`: `[34, 44]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 34 [calculation_knowledge] "Safety Incident Score (SIS)"
    def: For a given robot R, SIS = \sum_{ps \in \text{performance_and_safety} \mid \text{effectivenessrobot = R}} (safety_metrics->>'overloads'::int + safety_metrics->>'collisions'::int + safety_metrics->>'emergency_stops'::int + safety_metrics->>'speed_violations'::int), \text{where JSONB fields are extrac…
  - KB 44 [domain_knowledge] "High Safety Concern"
    def: A robot R has High Safety Concern if SIS > 20.

### system_controller.row_controller_stress  (column)

- Current `meta.kb_ids`: `[36, 46]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 36 [calculation_knowledge] "Controller Stress Index (CSI)"
    def: For a given robot R, CSI = \frac{\sum_{sc \in \text{system_controller} \mid \text{systemoverseerrobot = R}} (controller_metrics->>'load_value'::float + controller_metrics->>'thermal_level'::float)}{\text{|}\{sc \in \text{system_controller} \mid \text{systemoverseerrobot = R}\}|}, \text{where JSONB f…
  - KB 46 [domain_knowledge] "Controller Overload Risk"
    def: A robot R has Controller Overload Risk if CSI > 100 and NO > 2.
