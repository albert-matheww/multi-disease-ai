# Worked examples (real held-out test rows; selection rule fixed in advance)

## Heart Disease

### first positive (test row 8, true label 1)

Inputs: age=35.0, sex=1.0, cp=4.0, trestbps=126.0, chol=282.0, fbs=0.0, restecg=2.0, thalach=156.0, exang=1.0, oldpeak=0.0, slope=1.0, ca=0.0, thal=7.0

- probability 0.727, band [0.065, 1.000], threshold 0.562, label **Heart Disease Present**, out-of-distribution: False
- top contributors: ca (-0.163); cp (+0.145); thal (+0.117)

### first negative (test row 0, true label 0)

Inputs: age=59.0, sex=1.0, cp=4.0, trestbps=138.0, chol=271.0, fbs=0.0, restecg=2.0, thalach=182.0, exang=0.0, oldpeak=0.0, slope=1.0, ca=0.0, thal=3.0

- probability 0.201, band [0.000, 0.864], threshold 0.562, label **No Heart Disease**, out-of-distribution: False
- top contributors: ca (-0.204); thal (-0.169); cp (+0.113)

### most novel (test row 15, true label 1)

Inputs: age=59.0, sex=1.0, cp=4.0, trestbps=164.0, chol=176.0, fbs=1.0, restecg=2.0, thalach=90.0, exang=0.0, oldpeak=1.0, slope=2.0, ca=2.0, thal=6.0

- probability 0.971, band [0.309, 1.000], threshold 0.562, label **Heart Disease Present**, out-of-distribution: True
- top contributors: ca (+0.139); cp (+0.086); thalach (+0.055)

## Diabetes

### first positive (test row 3, true label 1)

Inputs: pregnancies=7.0, glucose=114.0, blood_pressure=64.0, skin_thickness=nan, insulin=nan, bmi=27.4, diabetes_pedigree_function=0.732, age=34.0

- probability 0.303, band [0.000, 1.000], threshold 0.359, label **Non-Diabetic**, out-of-distribution: False
- top contributors: glucose_bmi_interaction (-0.105); diabetes_pedigree_function (+0.045); pregnancies (+0.038)

### first negative (test row 0, true label 0)

Inputs: pregnancies=7.0, glucose=159.0, blood_pressure=64.0, skin_thickness=nan, insulin=nan, bmi=27.4, diabetes_pedigree_function=0.294, age=40.0

- probability 0.631, band [0.000, 1.000], threshold 0.359, label **Diabetic**, out-of-distribution: False
- top contributors: glucose (+0.126); glucose_bmi_interaction (+0.071); age (+0.045)

### most novel (test row 144, true label 0)

Inputs: pregnancies=0.0, glucose=57.0, blood_pressure=60.0, skin_thickness=nan, insulin=nan, bmi=21.7, diabetes_pedigree_function=0.735, age=67.0

- probability 0.018, band [0.000, 0.733], threshold 0.359, label **Non-Diabetic**, out-of-distribution: True
- top contributors: glucose_bmi_interaction (-0.174); glucose (-0.109); bmi (-0.052)

## Chronic Kidney Disease

### first positive (test row 1, true label 1)

Inputs: age=65.0, bp=90.0, sg=1.01, al=4.0, su=2.0, rbc=normal, pc=normal, pcc=notpresent, ba=notpresent, bgr=172.0, bu=82.0, sc=13.5, sod=145.0, pot=6.3, hemo=8.8, pcv=31.0, wbcc=nan, rbcc=nan, htn=yes, dm=yes, cad=no, appet=good, pe=yes, ane=yes

- probability 1.000, band [1.000, 1.000], threshold 0.851, label **CKD Present**, out-of-distribution: False
- top contributors: hemo (+0.015); ane (+0.012); htn (+0.011)

### first negative (test row 0, true label 0)

Inputs: age=44.0, bp=60.0, sg=1.02, al=0.0, su=0.0, rbc=normal, pc=normal, pcc=notpresent, ba=notpresent, bgr=95.0, bu=46.0, sc=0.5, sod=138.0, pot=4.2, hemo=15.0, pcv=50.0, wbcc=7700.0, rbcc=6.3, htn=no, dm=no, cad=no, appet=good, pe=no, ane=no

- probability 0.000, band [0.000, 0.000], threshold 0.851, label **No CKD**, out-of-distribution: False
- top contributors: al (-0.221); sc (-0.193); comorbidity_count (-0.107)

### most novel (test row 56, true label 1)

Inputs: age=71.0, bp=60.0, sg=1.015, al=4.0, su=0.0, rbc=normal, pc=normal, pcc=notpresent, ba=notpresent, bgr=118.0, bu=125.0, sc=5.3, sod=136.0, pot=4.9, hemo=11.4, pcv=35.0, wbcc=15200.0, rbcc=4.3, htn=yes, dm=yes, cad=no, appet=poor, pe=yes, ane=no

- probability 1.000, band [1.000, 1.000], threshold 0.851, label **CKD Present**, out-of-distribution: False
- top contributors: dm (+0.020); htn (+0.014); comorbidity_count (+0.014)

## Liver Disease

### first positive (test row 2, true label 1)

Inputs: Age=66, Gender=Female, TB=4.2, DB=2.1, Alkphos=159, Sgpt=15, Sgot=30, TP=7.1, ALB=2.2, A/G Ratio=0.4

- probability 0.895, band [0.236, 1.000], threshold 0.752, label **Liver Disease Present**, out-of-distribution: False
- top contributors: DB (+0.051); ALB (+0.048); Sgpt (-0.044)

### first negative (test row 0, true label 0)

Inputs: Age=33, Gender=Male, TB=0.8, DB=0.2, Alkphos=135, Sgpt=30, Sgot=29, TP=7.2, ALB=4.4, A/G Ratio=1.5

- probability 0.503, band [0.000, 1.000], threshold 0.752, label **No Liver Disease**, out-of-distribution: False
- top contributors: ALB (-0.096); DB (-0.062); Alkphos (-0.053)

### most novel (test row 23, true label 1)

Inputs: Age=31, Gender=Male, TB=0.9, DB=0.2, Alkphos=518, Sgpt=189, Sgot=17, TP=5.3, ALB=2.3, A/G Ratio=0.7

- probability 0.741, band [0.083, 1.000], threshold 0.752, label **No Liver Disease**, out-of-distribution: True
- top contributors: Sgot (-0.063); Sgpt (+0.059); Alkphos (+0.051)

