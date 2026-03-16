Sentinel: Autonomous Livestock Intelligence and Forensic Audit



Sentinel is a computer vision framework designed to eliminate insurance fraud in the livestock industry. By utilizing a multi-agent approach, the system verifies Proof of Life and Proof of Identity through anatomical biometrics and postural semantics, moving beyond the vulnerabilities of physical ear tags.



Key Innovations



Background-Agnostic Pipeline: Neutralizes environmental straw bias using YOLOv11 and OpenCV GrabCut masking to isolate anatomical features.



Biometric Identity Lock: Implements a Siamese Network (ResNet-18 backbone) to compare live registration photos with mortality claims, achieving a high-confidence identity verdict.



Geometric Posture Guard: Uses mathematical aspect ratio analysis to verify a prostrate (deceased) position, providing a fail-safe against staged mortality fraud.



Edge-AI Optimized: Specifically engineered for high-efficiency inference on Intel Core i3 CPU-only hardware for rural deployment.



Performance Metrics



Primary Metric: mAP50: 0.941



Loss Convergence: Box Loss: 0.012



Latency: Optimized from 2.2s to 0.05s per audit via refactored preprocessing loops.



Training Milestone: 150-epoch cycle completed in 17.5 hours on CPU.



Repository Structure



Sentinel\_Main\_Research.ipynb: The complete research workflow from data cleaning to model training.



identity\_test.py: Standalone integration test to verify the biometric matching reliability.



requirements.txt: Python dependencies required to replicate the environment.



sentinel\_bridge.py: The backend API connecting the AI models to the Audit Portal.



Installation



Clone the repository.



Ensure you have Python 3.10+ installed.



Install dependencies:



pip install -r requirements.txt







Run the integration test to verify biometric logic:



python identity\_test.py







Author: Drishti Adlakha. Research for Paris-Saclay Evaluation and IEEE ICDL (Kyoto, Japan).

## Contact
Drishti Adlakha - [LinkedIn](https://www.linkedin.com/in/drishtia/)
