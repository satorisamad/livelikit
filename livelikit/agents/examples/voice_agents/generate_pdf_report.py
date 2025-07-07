#!/usr/bin/env python3
"""
Generate PDF Report from Comprehensive Voice Agent Report
Converts the markdown report to a professional PDF document
"""

import os
import sys
from datetime import datetime
from pathlib import Path

def install_requirements():
    """Install required packages for PDF generation"""
    try:
        import markdown
        import pdfkit
        import weasyprint
    except ImportError:
        print("Installing required packages...")
        os.system("pip install markdown pdfkit weasyprint")
        print("Packages installed successfully!")

def markdown_to_html(markdown_file):
    """Convert markdown to HTML with styling"""
    try:
        import markdown
        from markdown.extensions import codehilite, tables, toc
    except ImportError:
        install_requirements()
        import markdown
        from markdown.extensions import codehilite, tables, toc
    
    # Read markdown content
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Configure markdown extensions
    extensions = [
        'codehilite',
        'tables',
        'toc',
        'fenced_code',
        'attr_list'
    ]
    
    # Convert to HTML
    md = markdown.Markdown(extensions=extensions)
    html_content = md.convert(markdown_content)
    
    # Add professional CSS styling
    css_style = """
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            font-size: 2.5em;
            margin-top: 0;
        }
        
        h2 {
            color: #34495e;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
            margin-top: 30px;
            font-size: 1.8em;
        }
        
        h3 {
            color: #2c3e50;
            margin-top: 25px;
            font-size: 1.4em;
        }
        
        h4 {
            color: #34495e;
            margin-top: 20px;
            font-size: 1.2em;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        tr:hover {
            background-color: #e8f4f8;
        }
        
        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            color: #e74c3c;
        }
        
        pre {
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 15px 0;
        }
        
        pre code {
            background-color: transparent;
            color: #ecf0f1;
            padding: 0;
        }
        
        blockquote {
            border-left: 4px solid #3498db;
            margin: 15px 0;
            padding: 10px 20px;
            background-color: #f8f9fa;
            font-style: italic;
        }
        
        .executive-summary {
            background-color: #e8f6f3;
            border: 1px solid #27ae60;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .key-finding {
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }
        
        .success {
            color: #27ae60;
            font-weight: bold;
        }
        
        .warning {
            color: #f39c12;
            font-weight: bold;
        }
        
        .error {
            color: #e74c3c;
            font-weight: bold;
        }
        
        .metric {
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 10px;
            margin: 10px 0;
        }
        
        .page-break {
            page-break-before: always;
        }
        
        .header-info {
            text-align: right;
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 30px;
        }
        
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }
        
        @media print {
            body {
                margin: 0;
                padding: 15px;
            }
            
            .page-break {
                page-break-before: always;
            }
        }
    </style>
    """
    
    # Create complete HTML document
    html_document = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Comprehensive Voice Agent Performance Report</title>
        {css_style}
    </head>
    <body>
        <div class="header-info">
            Generated on: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
        </div>
        {html_content}
        <div class="footer">
            <p>Comprehensive Voice Agent Performance Report | LiveKit Implementation</p>
            <p>Generated from comprehensive analysis of voice agent optimization efforts</p>
        </div>
    </body>
    </html>
    """
    
    return html_document

def html_to_pdf(html_content, output_file):
    """Convert HTML to PDF using weasyprint"""
    try:
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
        
        print("Converting HTML to PDF using WeasyPrint...")
        
        # Create PDF
        html_doc = HTML(string=html_content)
        html_doc.write_pdf(output_file)
        
        print(f"✅ PDF generated successfully: {output_file}")
        return True
        
    except ImportError:
        print("WeasyPrint not available, trying pdfkit...")
        return html_to_pdf_pdfkit(html_content, output_file)
    except Exception as e:
        print(f"❌ Error with WeasyPrint: {e}")
        return html_to_pdf_pdfkit(html_content, output_file)

def html_to_pdf_pdfkit(html_content, output_file):
    """Fallback: Convert HTML to PDF using pdfkit"""
    try:
        import pdfkit
        
        print("Converting HTML to PDF using pdfkit...")
        
        # Configure pdfkit options
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None
        }
        
        # Generate PDF
        pdfkit.from_string(html_content, output_file, options=options)
        print(f"✅ PDF generated successfully: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        print("💡 Try installing wkhtmltopdf: https://wkhtmltopdf.org/downloads.html")
        return False

def main():
    """Main function to generate PDF report"""
    print("🚀 Voice Agent Report PDF Generator")
    print("=" * 50)
    
    # File paths
    script_dir = Path(__file__).parent
    markdown_file = script_dir / "COMPREHENSIVE_VOICE_AGENT_REPORT.md"
    html_file = script_dir / "report.html"
    pdf_file = script_dir / "Voice_Agent_Performance_Report.pdf"
    
    # Check if markdown file exists
    if not markdown_file.exists():
        print(f"❌ Markdown file not found: {markdown_file}")
        return False
    
    print(f"📄 Reading markdown file: {markdown_file}")
    
    try:
        # Convert markdown to HTML
        print("🔄 Converting markdown to HTML...")
        html_content = markdown_to_html(markdown_file)
        
        # Save HTML file (for debugging)
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"💾 HTML file saved: {html_file}")
        
        # Convert HTML to PDF
        print("🔄 Converting HTML to PDF...")
        success = html_to_pdf(html_content, pdf_file)
        
        if success:
            print("\n🎉 Report generation completed successfully!")
            print(f"📊 PDF Report: {pdf_file}")
            print(f"🌐 HTML Version: {html_file}")
            print(f"📝 Source Markdown: {markdown_file}")
            
            # Show file sizes
            if pdf_file.exists():
                pdf_size = pdf_file.stat().st_size / 1024  # KB
                print(f"📏 PDF Size: {pdf_size:.1f} KB")
            
            return True
        else:
            print("\n❌ Failed to generate PDF")
            print(f"🌐 HTML version available: {html_file}")
            return False
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
