#!/usr/bin/env python3
"""
Simple PDF Generator for Voice Agent Report
Uses reportlab for reliable PDF generation without external dependencies
"""

import os
import sys
from datetime import datetime
from pathlib import Path

def install_reportlab():
    """Install reportlab for PDF generation"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        print("✅ ReportLab already installed")
        return True
    except ImportError:
        print("📦 Installing ReportLab...")
        os.system("pip install reportlab")
        print("✅ ReportLab installed successfully!")
        return True

def parse_markdown_content(markdown_file):
    """Parse markdown content and extract structured data"""
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    sections = []
    current_section = None
    current_content = []
    
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('# '):
            # Main title
            if current_section:
                sections.append({
                    'type': current_section['type'],
                    'title': current_section['title'],
                    'content': '\n'.join(current_content)
                })
            current_section = {'type': 'title', 'title': line[2:]}
            current_content = []
            
        elif line.startswith('## '):
            # Section header
            if current_section:
                sections.append({
                    'type': current_section['type'],
                    'title': current_section['title'],
                    'content': '\n'.join(current_content)
                })
            current_section = {'type': 'section', 'title': line[3:]}
            current_content = []
            
        elif line.startswith('### '):
            # Subsection
            if current_section:
                sections.append({
                    'type': current_section['type'],
                    'title': current_section['title'],
                    'content': '\n'.join(current_content)
                })
            current_section = {'type': 'subsection', 'title': line[4:]}
            current_content = []
            
        elif line.startswith('| ') and '|' in line:
            # Table row
            current_content.append(line)
            
        else:
            # Regular content
            if line:
                current_content.append(line)
    
    # Add the last section
    if current_section:
        sections.append({
            'type': current_section['type'],
            'title': current_section['title'],
            'content': '\n'.join(current_content)
        })
    
    return sections

def parse_table(table_lines):
    """Parse markdown table into data structure"""
    if not table_lines:
        return None
    
    # Find header and data rows
    header_row = None
    data_rows = []
    
    for line in table_lines:
        if line.startswith('| ') and line.endswith(' |'):
            cells = [cell.strip() for cell in line[1:-1].split('|')]
            if header_row is None and not all(cell.startswith('-') for cell in cells):
                header_row = cells
            elif not all(cell.startswith('-') for cell in cells):
                data_rows.append(cells)
    
    if header_row and data_rows:
        return {'header': header_row, 'data': data_rows}
    return None

def create_pdf_report(sections, output_file):
    """Create PDF report using ReportLab"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    except ImportError:
        install_reportlab()
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    
    # Create document (convert Path to string)
    doc = SimpleDocTemplate(str(output_file), pagesize=A4,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    section_style = ParagraphStyle(
        'CustomSection',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkblue
    )
    
    subsection_style = ParagraphStyle(
        'CustomSubsection',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=15,
        textColor=colors.darkgreen
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    )
    
    # Build story
    story = []
    
    # Add header info
    header_text = f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
    story.append(Paragraph(header_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    for section in sections:
        if section['type'] == 'title':
            story.append(Paragraph(section['title'], title_style))
            story.append(Spacer(1, 20))
            
        elif section['type'] == 'section':
            story.append(Paragraph(section['title'], section_style))
            
        elif section['type'] == 'subsection':
            story.append(Paragraph(section['title'], subsection_style))
        
        # Process content
        content = section['content']
        if content:
            # Check if content contains tables
            lines = content.split('\n')
            table_lines = []
            text_lines = []
            
            for line in lines:
                if line.startswith('| ') and '|' in line:
                    table_lines.append(line)
                else:
                    if table_lines:
                        # Process accumulated table
                        table_data = parse_table(table_lines)
                        if table_data:
                            story.append(create_table(table_data))
                            story.append(Spacer(1, 12))
                        table_lines = []
                    
                    if line.strip():
                        text_lines.append(line)
            
            # Process any remaining table
            if table_lines:
                table_data = parse_table(table_lines)
                if table_data:
                    story.append(create_table(table_data))
                    story.append(Spacer(1, 12))
            
            # Add text content
            if text_lines:
                text_content = '\n'.join(text_lines)
                # Split into paragraphs
                paragraphs = text_content.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        # Clean up markdown formatting
                        clean_para = para.replace('**', '').replace('*', '').replace('`', '')
                        story.append(Paragraph(clean_para, body_style))
                        story.append(Spacer(1, 6))
        
        story.append(Spacer(1, 12))
    
    # Add footer
    story.append(PageBreak())
    footer_text = "Comprehensive Voice Agent Performance Report | LiveKit Implementation"
    story.append(Paragraph(footer_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    return True

def create_table(table_data):
    """Create a formatted table from table data"""
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors
    
    # Prepare data
    data = [table_data['header']] + table_data['data']
    
    # Create table
    table = Table(data)
    
    # Style table
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    return table

def main():
    """Main function to generate PDF report"""
    print("🚀 Simple Voice Agent PDF Generator")
    print("=" * 50)
    
    # File paths
    script_dir = Path(__file__).parent
    markdown_file = script_dir / "COMPREHENSIVE_VOICE_AGENT_REPORT.md"
    pdf_file = script_dir / "Voice_Agent_Performance_Report_Simple.pdf"
    
    # Check if markdown file exists
    if not markdown_file.exists():
        print(f"❌ Markdown file not found: {markdown_file}")
        return False
    
    print(f"📄 Reading markdown file: {markdown_file}")
    
    try:
        # Parse markdown content
        print("🔄 Parsing markdown content...")
        sections = parse_markdown_content(markdown_file)
        print(f"📊 Found {len(sections)} sections")
        
        # Create PDF
        print("🔄 Creating PDF report...")
        success = create_pdf_report(sections, pdf_file)
        
        if success:
            print("\n🎉 PDF Report generated successfully!")
            print(f"📊 PDF Report: {pdf_file}")
            
            # Show file size
            if pdf_file.exists():
                pdf_size = pdf_file.stat().st_size / 1024  # KB
                print(f"📏 PDF Size: {pdf_size:.1f} KB")
            
            return True
        else:
            print("\n❌ Failed to generate PDF")
            return False
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
