"""
PDF Export Module
Generates PDF reports from plan data with graphs and task information.
Uses reportlab for better table handling and alignment.
"""
import os
import re
import pandas as pd
from datetime import datetime
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
from meeting_notes import get_note_for_date, get_all_notes

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas

def generate_pdf_report(
    plan,
    all_ranges,
    plot_effort_graphs_func,
    get_active_task_by_label_func,
    max_effort_func,
    use_filters=False,
    selected_ranges=None,
    selected_labels=None,
    selected_countries=None,
    selected_projects=None,
    analysis_date=None,
    max_capacity_values=None,
    excel_filename=None
):
    """
    Generate a PDF report from plan data using reportlab.
    
    Args:
        plan: Plan object
        all_ranges: Dictionary of all date ranges
        plot_effort_graphs_func: Function to generate effort graphs (not used, we use matplotlib)
        get_active_task_by_label_func: Function to get active tasks by label
        max_effort_func: Function to get max effort for a label
        use_filters: Whether to apply filters
        selected_ranges: List of selected range titles (if use_filters)
        selected_labels: List of selected labels (if use_filters)
        selected_countries: List of selected countries (if use_filters)
        selected_projects: List of selected projects (if use_filters)
        analysis_date: Date object for the analysis
        max_capacity_values: Dict of max capacity values
    
    Returns:
        bytes: PDF file as bytes
    """
    # Determine which ranges to use
    if use_filters and selected_ranges:
        ranges = {title: all_ranges[title] for title in selected_ranges if title in all_ranges}
    else:
        ranges = all_ranges
    
    # Create PDF in memory
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    # Container for the 'Flowable' objects
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=12,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=10,
        spaceBefore=10
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=8,
        spaceBefore=8
    )
    
    label_style = ParagraphStyle(
        'LabelStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=5,
        spaceBefore=5
    )
    
    # Title
    title = Paragraph('Engineering Plan Review - 3 Week Report', title_style)
    elements.append(title)
    
    # Add Excel filename and analysis date
    info_lines = []
    if excel_filename:
        info_lines.append(f'Data Source: {excel_filename}')
    if analysis_date:
        info_lines.append(f'Analysis Date: {analysis_date.strftime("%Y-%m-%d")}')
    
    if info_lines:
        info_text = ' | '.join(info_lines)
        date_text = Paragraph(info_text, styles['Normal'])
        elements.append(date_text)
        elements.append(Spacer(1, 0.3*inch))
    
    # Add last 5 meeting notes at the start
    try:
        all_notes = get_all_notes()
        if all_notes:
            # Sort notes by date (most recent first)
            sorted_note_dates = sorted(all_notes.keys(), reverse=True)
            # Get last 5 (most recent)
            recent_note_dates = sorted_note_dates[:5]
            
            if recent_note_dates:
                # Meeting notes section header
                notes_section_heading = Paragraph('Recent Meeting Notes', heading_style)
                elements.append(notes_section_heading)
                elements.append(Spacer(1, 0.1*inch))
                
                # Add each note
                for note_date_str in recent_note_dates:
                    note_date = datetime.strptime(note_date_str, '%Y-%m-%d').date()
                    note_content = all_notes[note_date_str]
                    
                    if note_content and note_content.strip():
                        # Date header for this note
                        date_heading = Paragraph(f'<b>{note_date_str}</b>', subheading_style)
                        elements.append(date_heading)
                        elements.append(Spacer(1, 0.05*inch))
                        
                        # Convert markdown to HTML
                        formatted_note = _convert_markdown_to_html(note_content)
                        
                        # Create styled paragraph
                        notes_style = ParagraphStyle(
                            'NotesStyle',
                            parent=styles['Normal'],
                            fontSize=9,
                            leading=12,
                            leftIndent=5,
                            rightIndent=5,
                            spaceAfter=5
                        )
                        
                        notes_para = Paragraph(formatted_note, notes_style)
                        
                        # Wrap in a table to create a bordered box
                        notes_table = Table([[notes_para]], colWidths=[9.5*inch])
                        notes_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8F9FA')),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                            ('LEFTPADDING', (0, 0), (-1, -1), 10),
                            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                            ('TOPPADDING', (0, 0), (-1, -1), 10),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#D5D8DC')),
                        ]))
                        
                        elements.append(notes_table)
                        elements.append(Spacer(1, 0.15*inch))
                
                # Add page break after meeting notes
                elements.append(PageBreak())
    except Exception as e:
        # If there's an error loading notes, just continue
        pass
    
    # Process each week range separately
    import tempfile
    graph_paths = []
    range_list = list(ranges.items())
    
    for week_index, (title, (start, end)) in enumerate(range_list):
        # Week section header
        week_heading = Paragraph(title, heading_style)
        elements.append(week_heading)
        elements.append(Spacer(1, 0.2*inch))
        
        # Generate graph for this week
        try:
            tasks = plan.get_all_tasks()
            effort_data = _get_daily_effort_by_label(tasks, start, end, max_effort_func)
            
            # Filter by labels if labels are selected
            if use_filters and selected_labels:
                effort_data = {label: effort_data[label] for label in effort_data.keys() if label in selected_labels}
            
            if effort_data:
                num_days = (end - start).days + 1
                labels = list(effort_data.keys())
                values = list(effort_data.values())
                max_capacity = [max_effort_func(l) * num_days for l in labels]
                within_capacity = [min(c, m) for c, m in zip(values, max_capacity)]
                over_capacity = [max(0, c - m) for c, m in zip(values, max_capacity)]
                
                # Create single graph for this week
                fig, ax = plt.subplots(1, 1, figsize=(12, 5))
                
                # Create stacked bar chart
                ax.bar(labels, max_capacity, color='lightgrey', label='max effort')
                ax.bar(labels, within_capacity, color='blue', label='current effort')
                ax.bar(labels, over_capacity, bottom=within_capacity, color='red', label='overload')
                
                # Add max effort value labels above each max capacity bar
                for i, (label, max_cap) in enumerate(zip(labels, max_capacity)):
                    ax.text(i, max_cap, f'{max_cap:.0f}', 
                           ha='center', va='bottom', 
                           fontsize=9, fontweight='bold',
                           color='#2C3E50',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', 
                                   edgecolor='#7F8C8D',
                                   linewidth=1.5,
                                   alpha=0.9))
                
                ax.set_title(title, fontsize=12)
                ax.set_ylabel('Effort (hours)')
                ax.set_xlabel('Label')
                ax.tick_params(axis='x', rotation=45)
                ax.legend(fontsize=9)
                
                plt.tight_layout()
                
                # Save to temp file
                tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
                plt.savefig(tmp_path, dpi=150, bbox_inches='tight')
                plt.close()
                graph_paths.append(tmp_path)
                
                # Add graph to PDF
                img = Image(tmp_path, width=9*inch, height=4*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.2*inch))
        except Exception as e:
            error_text = Paragraph(f'Error generating graph: {str(e)}', styles['Normal'])
            elements.append(error_text)
            elements.append(Spacer(1, 0.1*inch))
        
        # Get tasks for this week, grouped by country then by label
        weekly_tasks = get_active_task_by_label_func(plan.get_all_tasks(), start, end)
        tasks_by_country_label = defaultdict(lambda: defaultdict(list))
        tasks_seen = set()
        
        # Countries list
        countries = ['Chile', 'Brasil', 'Mexico', 'Colombia']
        if use_filters and selected_countries:
            countries = [c for c in countries if c in selected_countries]
        
        for label, tasks in weekly_tasks.items():
            # Filter by label if labels are selected
            if use_filters and selected_labels and label not in selected_labels:
                continue
            
            for task in tasks:
                # Skip summary tasks and completed tasks
                if task.is_summary or task.completed:
                    continue
                
                # Filter by country if countries are selected
                if use_filters and selected_countries:
                    if pd.isna(task.country) or task.country not in selected_countries:
                        continue
                
                # Filter by project if projects are selected
                if use_filters and selected_projects:
                    if pd.isna(task.project) or task.project not in selected_projects:
                        continue
                
                # Get country (default to 'No Country' if missing)
                country = task.country if not pd.isna(task.country) and task.country else 'No Country'
                
                # Only include if country is in our list
                if country in countries or country == 'No Country':
                    # Add task to country->label group (avoid duplicates)
                    task_id = id(task)
                    if task_id not in tasks_seen:
                        tasks_seen.add(task_id)
                        tasks_by_country_label[country][label].append(task)
        
        # Create task tables organized by country, then by label
        has_tasks = any(tasks_by_country_label[country] for country in tasks_by_country_label.keys())
        
        if has_tasks:
            # Iterate through countries
            for country in countries:
                country_tasks = tasks_by_country_label.get(country, {})
                
                # Check if country has any tasks
                if not country_tasks:
                    continue
                
                # Country header
                country_heading = Paragraph(country, subheading_style)
                elements.append(country_heading)
                elements.append(Spacer(1, 0.1*inch))
                
                # Sort labels alphabetically for this country
                sorted_labels = sorted(country_tasks.keys())
                
                for label in sorted_labels:
                    tasks = country_tasks[label]
                    
                    if not tasks:
                        continue
                    
                    # Label header
                    label_text = Paragraph(f'<b>Label: {label}</b>', label_style)
                    elements.append(label_text)
                    elements.append(Spacer(1, 0.05*inch))
                    
                    # Sort tasks by project and due date
                    def sort_key(task):
                        project = str(task.project) if not pd.isna(task.project) else ''
                        due_date = task.due_date if task.due_date else datetime.max
                        return (project, due_date)
                    
                    sorted_tasks = sorted(tasks, key=sort_key)
                    
                    # Create table data
                    table_data = []
                    # Header row
                    table_data.append([
                        Paragraph('<b>Task Name</b>', styles['Normal']),
                        Paragraph('<b>Project</b>', styles['Normal']),
                        Paragraph('<b>Start Date</b>', styles['Normal']),
                        Paragraph('<b>Due Date</b>', styles['Normal']),
                        Paragraph('<b>Effort</b>', styles['Normal']),
                        Paragraph('<b>Status</b>', styles['Normal'])
                    ])
                    
                    # Data rows
                    for task in sorted_tasks:
                        # Task name
                        task_name = str(task.name) if task.name else ''
                        
                        # Project
                        project = str(task.project) if not pd.isna(task.project) and task.project else ''
                        
                        # Start date
                        start_date = task.start_date.strftime('%Y-%m-%d') if task.start_date else 'N/A'
                        
                        # Due date
                        due_date = task.due_date.strftime('%Y-%m-%d') if task.due_date else 'N/A'
                        
                        # Effort
                        effort = f"{task.effort:.1f}" if task.effort is not None and not pd.isna(task.effort) else 'N/A'
                        
                        # Status
                        if task.completed:
                            status = 'Done'
                            status_color = colors.HexColor('#27AE60')
                        elif task.is_overdue():
                            status = 'Overdue'
                            status_color = colors.HexColor('#E74C3C')
                        else:
                            status = 'Active'
                            status_color = colors.HexColor('#3498DB')
                        
                        # Create status paragraph with color
                        status_para = Paragraph(f'<font color="{status_color.hexval()}">{status}</font>', styles['Normal'])
                        
                        table_data.append([
                            Paragraph(task_name, styles['Normal']),
                            Paragraph(project, styles['Normal']),
                            Paragraph(start_date, styles['Normal']),
                            Paragraph(due_date, styles['Normal']),
                            Paragraph(effort, styles['Normal']),
                            status_para
                        ])
                    
                    # Create table
                    # Column widths for landscape: total width ~10 inches
                    col_widths = [3.5*inch, 2.0*inch, 1.2*inch, 1.2*inch, 1.0*inch, 0.8*inch]
                    
                    table = Table(table_data, colWidths=col_widths, repeatRows=1)
                    
                    # Table style
                    table.setStyle(TableStyle([
                        # Header row
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ECF0F1')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
                        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                        ('TOPPADDING', (0, 0), (-1, 0), 8),
                        
                        # Data rows
                        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                        ('ALIGN', (0, 1), (0, -1), 'LEFT'),  # Task name left
                        ('ALIGN', (1, 1), (1, -1), 'LEFT'),  # Project left
                        ('ALIGN', (2, 1), (3, -1), 'CENTER'),  # Dates center
                        ('ALIGN', (4, 1), (4, -1), 'RIGHT'),  # Effort right
                        ('ALIGN', (5, 1), (5, -1), 'CENTER'),  # Status center
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 1), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')]),
                    ]))
                    
                    elements.append(table)
                    elements.append(Spacer(1, 0.2*inch))
        else:
            # No tasks for this week
            no_tasks_text = Paragraph('No active tasks for this week.', styles['Normal'])
            elements.append(no_tasks_text)
            elements.append(Spacer(1, 0.1*inch))
        
        # Add page break between weeks (except after last week)
        if week_index < len(range_list) - 1:
            elements.append(PageBreak())
    
    # Build PDF
    doc.build(elements)
    
    # Clean up temp files
    for tmp_path in graph_paths:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    # Get PDF bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def _get_daily_effort_by_label(tasks, from_date, to_date, max_effort_func, seen=None):
    """Calculate daily effort by label for a date range."""
    effort_by_label = defaultdict(float)
    
    if seen is None:
        seen = set()
    
    for task in tasks:
        if id(task) in seen:
            continue
        seen.add(id(task))
        
        if task.is_summary:
            subtasks_effort = _get_daily_effort_by_label(task.subtasks, from_date, to_date, max_effort_func, seen)
            for label, value in subtasks_effort.items():
                effort_by_label[label] += value
        else:
            if task.daily_effort and task.due_date and _active_during_analysis_period(task, from_date, to_date):
                label = task.label or 'No Label'
                if pd.isna(label):
                    label = 'No Label'
                subtotal_effort = float(task.daily_effort) * ((to_date - from_date).days + 1)
                if pd.isna(subtotal_effort):
                    subtotal_effort = 0
                effort_by_label[label] += subtotal_effort
    return effort_by_label


def _active_during_analysis_period(task, from_date, to_date):
    """Check if task is active during the analysis period."""
    task_start = task.start_date
    task_finish = task.due_date
    active = False
    if task_start is None or task_finish is None:
        active = False
    else:
        if not task.completed:
            if datetime.today() <= to_date and datetime.today() >= from_date:
                active = task_start <= to_date
            else:
                active = task_start <= to_date and task_finish >= from_date and not task.completed
    return active


def _convert_markdown_to_html(text):
    """Convert basic markdown formatting to HTML for reportlab Paragraph."""
    if not text:
        return ""
    
    # Escape HTML special characters first
    import html
    text = html.escape(text)
    
    # Convert markdown to HTML
    # Headers
    lines = text.split('\n')
    result_lines = []
    
    for line in lines:
        # Headers (# ## ###)
        if line.startswith('### '):
            result_lines.append(f'<b><font size="12">{line[4:]}</font></b>')
        elif line.startswith('## '):
            result_lines.append(f'<b><font size="13">{line[3:]}</font></b>')
        elif line.startswith('# '):
            result_lines.append(f'<b><font size="14">{line[2:]}</font></b>')
        # Bold (**text** or __text__)
        elif '**' in line or '__' in line:
            # Replace **text** with <b>text</b>
            line = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
            line = re.sub(r'__(.+?)__', r'<b>\1</b>', line)
            # Replace *text* with <i>text</i> (but not **text**)
            line = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', line)
            result_lines.append(line)
        # Bullet lists (- or *)
        elif line.strip().startswith('- ') or line.strip().startswith('* '):
            content = line.strip()[2:]
            result_lines.append(f'&bull; {content}')
        # Numbered lists
        elif re.match(r'^\d+\.\s', line.strip()):
            result_lines.append(line)
        # Horizontal rule
        elif line.strip() == '---' or line.strip() == '***':
            result_lines.append('<hr/>')
        # Code blocks (backticks)
        elif '`' in line:
            line = re.sub(r'`([^`]+)`', r'<font face="Courier"><b>\1</b></font>', line)
            result_lines.append(line)
        # Links [text](url)
        elif '[' in line and '](' in line:
            line = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'<u>\1</u>', line)
            result_lines.append(line)
        # Checkboxes
        elif line.strip().startswith('- [ ]'):
            content = line.strip()[5:]
            result_lines.append(f'☐ {content}')
        elif line.strip().startswith('- [x]') or line.strip().startswith('- [X]'):
            content = line.strip()[5:]
            result_lines.append(f'☑ {content}')
        else:
            result_lines.append(line)
    
    # Join with <br/> for line breaks
    result = '<br/>'.join(result_lines)
    
    # Clean up multiple consecutive breaks
    result = re.sub(r'(<br/>){3,}', '<br/><br/>', result)
    
    return result
