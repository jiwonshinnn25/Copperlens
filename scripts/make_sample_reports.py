"""Generate SYNTHETIC mock daily metals reports (PDF) for demoing the NLP desk.

All figures and events are fictional - they exist only to exercise the pipeline.
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "sample_reports"

REPORTS = {
    "mock_report_bullish.pdf": (
        "Daily Metals Brief - Copper tightens on supply disruption",
        [
            "Copper rallied for a third session as a strike at a major Chilean mine raised fears of a supply disruption lasting several weeks.",
            "Exchange inventories fell for the fifth straight day, and visible stocks are now at their lowest level this year, a sign of tightening physical supply.",
            "Demand growth remained firm, with grid investment orders and EV demand supporting restocking by Asian fabricators. China's PMI rose above 50 for the second month.",
            "The dollar fell to a two-week low after dovish central bank commentary, a weaker dollar that lifts metals priced in USD.",
            "Technically, the market is testing resistance near $5.15/lb; a breakout above this level would target $5.40/lb, while support sits at $4.95/lb.",
        ],
    ),
    "mock_report_bearish.pdf": (
        "Daily Metals Brief - Copper slips as stocks build and demand slows",
        [
            "Copper fell for a second day as inventories rose for the fourth week, with visible stocks climbing to a six-month high.",
            "Analysts flagged weak demand from the construction sector, citing a continued property slump and a manufacturing slowdown in Europe.",
            "A stronger dollar and hawkish rate expectations weighed on the complex, while risk-off sentiment hit speculative positioning.",
            "New tariff announcements raised concerns about trade flows, and record output from two large mines points to a growing surplus this quarter.",
            "Support is seen at $4.60/lb; a break lower could open a move toward $4.40/lb, with resistance at $4.85/lb.",
        ],
    ),
    "mock_report_neutral.pdf": (
        "Daily Metals Brief - Copper range-bound ahead of data",
        [
            "Copper traded in a narrow range as investors awaited inflation data and a central bank decision later this week.",
            "Inventories were little changed, and a minor outage at one smelter was offset by a modest ramp-up at another operation.",
            "Demand signals were mixed: restocking in Asia was reported, but a slowdown in some European orders tempered optimism.",
            "The market holds between support at $4.75/lb and resistance at $4.95/lb.",
        ],
    ),
}

styles = getSampleStyleSheet()
for fname, (title, paras) in REPORTS.items():
    doc = SimpleDocTemplate(str(OUT / fname), pagesize=A4, title=title)
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    for p in paras:
        story += [Paragraph(p, styles["BodyText"]), Spacer(1, 8)]
    story += [Spacer(1, 16), Paragraph("<i>SYNTHETIC SAMPLE - not real market commentary.</i>", styles["BodyText"])]
    doc.build(story)
    print("wrote", OUT / fname)
