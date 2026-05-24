# W4d: crypto

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### effective_leverage.effective_leverage  (model)

- Current `meta.kb_ids`: `[33, 52]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 33 [calculation_knowledge] "Effective Leverage"
    def: Effective Leverage = posmagn \times \frac{possum}{walletsum}, \text{where } posmagn \text{ is the position leverage, } possum \text{ is the notional value of position, and } walletsum \text{ is the total wallet balance.}
  - KB 52 [domain_knowledge] "Effective Leverage Risk Classification"
    def: A position is labeled as 'High Risk' if its Effective Leverage exceeds 20, otherwise as 'Normal'.

### margin_utilization.margin_utilization  (model)

- Current `meta.kb_ids`: `[7, 18]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 7 [calculation_knowledge] "Margin Utilization"
    def: Margin Utilization = \frac{inithold}{margsum} \times 100, \text{where } inithold \text{ is the initial margin required and } margsum \text{ is the margin account balance.}
  - KB 18 [domain_knowledge] "Margin Call Risk"
    def: Accounts where the Margin Utilization exceeds 80%, putting them at risk of margin calls if market prices move adversely.

### market_depth_ratio.market_depth_ratio  (model)

- Current `meta.kb_ids`: `[36, 43]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 36 [calculation_knowledge] "Market Depth Ratio"
    def: Market Depth Ratio = \frac{biddepth \text{ or } askdepth}{dealcount} \times Liquidity Ratio, \text{where } biddepth/askdepth \text{ is used depending on position direction (posedge), } dealcount \text{ is the order quantity, and } Liquidity Ratio \text{ measures available liquidity to total market v…
  - KB 43 [domain_knowledge] "Liquidity Constrained Position"
    def: A position where the Market Depth Ratio is less than 2.0, indicating that the position size is large relative to available market depth, potentially leading to significant slippage upon exit.

### realized_risk_ratio.realized_risk_ratio  (model)

- Current `meta.kb_ids`: `[6, 30]`
- Bucket: **D** — R-SPLIT-MULTI-FORMULA (one entity per formula)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 6 [calculation_knowledge] "Realized Risk Ratio (RRR)"
    def: RRR = \frac{realline}{PVaR}, \text{where } realline \text{ is the realized PnL and } PVaR \text{ is the Position Value at Risk.}
  - KB 30 [calculation_knowledge] "Risk-Adjusted Return"
    def: Risk-Adjusted Return = \frac{realline}{PVaR \times posriskrate}, \text{where } realline \text{ is the realized PnL, } PVaR \text{ is the Position Value at Risk, and } posriskrate \text{ is the position risk ratio.}
