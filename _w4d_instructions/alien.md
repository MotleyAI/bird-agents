# W4d: alien

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### signal_full.signal_full  (model)

- Current `meta.kb_ids`: `[6, 10, 11, 18, 19, 24, 31, 33, 34, 36, 37, 40, 41, 42, 43, 45, 46, 47, 48, 49, 54]`
- Bucket: **E** — R-SPLIT-MONSTER (aggressive split; agent judgment required)
- Target: produce 21 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 6 [calculation_knowledge] "Encoding Complexity Index (ECI)"
    def: $\text{ECI} = \frac{\text{CompressRatio} \times \text{ComplexIdx} \times \text{EntropyVal}}{10}$, where values above 1.5 suggest deliberate information encoding rather than random patterns.
  - KB 10 [domain_knowledge] "Technosignature"
    def: A signal with $\text{TechSigProb} > 0.7$, $\text{NatSrcProb} < 0.3$, and $\text{ArtSrcProb} < 50$ that exhibits narrow bandwidth ($\text{BFR} < 0.001$) and high information density ($\text{InfoDense} > 0.8$).
  - KB 11 [domain_knowledge] "Coherent Information Pattern (CIP)"
    def: Signals characterized by high signal stability ($\text{SSM} > 0.8$), organized information structure ($\text{EntropyVal}$ between 0.4-0.8), and consistent modulation ($\text{ModType}$ with $\text{ModIndex} > 0.5$).
  - KB 18 [domain_knowledge] "Encoded Information Transfer (EIT)"
    def: Signals with $\text{ECI} > 1.8$, $\text{EntropyVal}$ between 0.3-0.7 (not random but structured), and consistent internal patterns that suggest language or data encoding schemes.
  - KB 19 [domain_knowledge] "Fast Radio Transient (FRT)"
    def: Signals with extremely short duration ($\text{SigDurSec} < 0.1$), high signal strength ($\text{SigStrDb} > 15$), broad bandwidth ($\text{BwHz} > 1000000$), and no periodicity ($\text{RepeatCount} = 1$).
  - KB 24 [domain_knowledge] "CIP Classification Label"
    def: Classification labels: 'Coherent Information Pattern Detected' ($\text{SSM} > 0.8$, $\text{EntropyVal}$ between 0.4-0.8, and $\text{ModIndex} > 0.5$), 'Potential Information Pattern' ($\text{SSM} > 0.6$ and $\text{EntropyVal}$ between 0.3-0.9$), or 'No Clear Pattern' (all other signals).
  - KB 31 [calculation_knowledge] "Artificial Intelligence Detection Probability (AIDP)"
    def: $\text{AIDP} = \frac{\text{ECI} \times \text{TOLS}}{1 + \text{NatSrcProb}}$, where ECI (Encoding Complexity Index) and TOLS (Technological Origin Likelihood Score) are weighted against natural source probability.
  - KB 33 [calculation_knowledge] "Information Entropy Ratio (IER)"
    def: $\text{IER} = \frac{\text{EntropyVal}}{\text{NatSrcProb} \times 0.9 + 0.1}$, where values significantly greater than 1 suggest non-natural information content. Uses NatSrcProb as a baseline for expected natural entropy.
  - KB 34 [calculation_knowledge] "Signal Processing Efficiency Index (SPEI)"
    def: $\text{SPEI} = \frac{\text{DecodeIters} \times \text{ProcTimeHrs}}{\text{ECI} \times \text{ComplexIdx}}$, where ECI (Encoding Complexity Index) provides the complexity component to normalize processing time and iterations.
  - KB 36 [calculation_knowledge] "Confirmation Confidence Score (CCS)"
    def: $\text{CCS} = (1 - \text{FalsePosProb}) \times \text{DecodeConf} \times \text{ClassConf} \times (\text{SNQI} > 0 ? \frac{\text{SNQI}}{10} + 0.5 : 0.1)$, where SNQI (Signal-to-Noise Quality Indicator) provides a quality weighting factor.
  - KB 37 [calculation_knowledge] "Habitable Zone Signal Relevance (HZSR)"
    def: $\text{HZSR} = \text{TOLS} \times (\text{ObjType} == \text{'Dwarf'} ? (0.7 \leq \text{ObjMassSol} \leq 1.4 ? (0.8 \leq \frac{\text{SourceDistLy}}{\sqrt{\text{ObjMassSol}}} \leq 1.7 ? 2 : 0.5) : 0.3) : 0.1)$, where TOLS (Technological Origin Likelihood Score) is weighted by stellar habitability facto…
  - KB 40 [domain_knowledge] "High-Confidence Technosignature"
    def: A Technosignature with $\text{CCS} > 0.9$, $\text{MCS} > 1.5$, and $\text{AIDP} > 0.8$, indicating a signal that meets the basic Technosignature criteria with additional confirmation through modulation complexity and artificial intelligence detection markers.
  - KB 41 [domain_knowledge] "Habitable Zone Transmission"
    def: A signal with $\text{HZSR} > 1.5$ and Technosignature characteristics, originating from a star system with conditions potentially suitable for life, making it a priority candidate for SETI research.
  - KB 42 [domain_knowledge] "Multi-Channel Communication Protocol"
    def: Signal exhibiting Coherent Information Pattern (CIP) characteristics across multiple frequency channels with coordinated timing ($\text{RepeatCount} > 3$, $\text{PeriodSec}$ consistent across observations) and $\text{ECI} > 2.0$, suggesting a designed communication system.
  - KB 43 [domain_knowledge] "Quantum-Coherent Transmission"
    def: Signals with $\text{QuantEffects}$ containing 'Significant' or 'Observed' patterns, exhibiting unusually high information density ($\text{InfoDense} > 1.5$) while maintaining an $\text{ECI} > 2.5$, suggesting advanced transmission technologies beyond conventional radiofrequency methods.
  - KB 45 [domain_knowledge] "Directed Transmission"
    def: Signals with high spatial stability ($\text{SpatStab} = \text{'Moderate'}$), narrow beam characteristics ($\text{PolarMode} = \text{'Linear'}$ with stable $\text{PolarAngleDeg}$), and high $\text{TOLS} > 0.85$, suggesting intentional transmission toward our location.
  - KB 46 [domain_knowledge] "Signal of Galactic Significance"
    def: Signals originating from regions of high $\text{CLSF} (> 2.0)$ that display Technosignature characteristics and have $\text{AIDP} > 0.7$, representing potential evidence of advanced civilizations at galactic-relevant locations.
  - KB 47 [calculation_knowledge] "CCS Approximation"
    def: $(1 - \text{FalsePosProb}) \times \text{DecodeConf} \times (\text{SNR} - 0.1 \times |\text{NoiseFloorDbm}| > 0 ? \frac{\text{SNR} - 0.1 \times |\text{NoiseFloorDbm}|}{10} + 0.5 : 0.1)$
  - KB 48 [domain_knowledge] "Observation-Verified Signal"
    def: Signals observed under Optimal Observing Window (OOW) conditions with $\text{OQF} > 0.85$ and $\text{CCS} > 0.8$, indicating high-quality observations with multiple verification methods applied.
  - KB 49 [domain_knowledge] "Anomalous Quantum Signal"
    def: Signals with $\text{QuantEffects}$ indicating anomalous behavior, $\text{AnomScore} > 8$, and unusually high $\text{MCS} (> 2.0)$, suggesting either unknown natural quantum phenomena or extremely advanced transmission technologies beyond current human capabilities.
  - KB 54 [domain_knowledge] "High Confidence Signals"
    def: Signals where $\text{CCS} > 0.8$
