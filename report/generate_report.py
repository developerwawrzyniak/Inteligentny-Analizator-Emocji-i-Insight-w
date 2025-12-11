from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


def generate(output="report.pdf", summary_text=""):
    doc = SimpleDocTemplate(output)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Summary", styles["Heading1"]),
        Paragraph(summary_text, styles["BodyText"]),
    ]
    doc.build(story)
