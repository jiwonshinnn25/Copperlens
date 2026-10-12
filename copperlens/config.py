"""Static configuration: tradable instruments and UI options."""

# label -> (Yahoo Finance futures symbol, unit label)
METALS = {
    "Copper":    ("HG=F", "USD/lb"),
    "Gold":      ("GC=F", "USD/oz"),
    "Silver":    ("SI=F", "USD/oz"),
    "Platinum":  ("PL=F", "USD/oz"),
    "Palladium": ("PA=F", "USD/oz"),
    "Aluminium": ("ALI=F", "USD/t"),
}

RANGES = {"6 months": "6mo", "1 year": "1y", "2 years": "2y"}

# Reference prices used ONLY by the offline GBM simulator (not real quotes).
SIM_BASE = {
    "HG=F": 4.80, "GC=F": 3400.0, "SI=F": 38.0,
    "PL=F": 1400.0, "PA=F": 1100.0, "ALI=F": 2700.0,
}
SIM_VOL = {  # annualised volatility used by the simulator
    "HG=F": 0.24, "GC=F": 0.15, "SI=F": 0.30,
    "PL=F": 0.26, "PA=F": 0.35, "ALI=F": 0.20,
}

MA_SHORT = 7
MA_LONG = 30
RSI_PERIOD = 14
