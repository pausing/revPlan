"""
PDF Export Module for Engineering Plan Review

This module handles the generation of PDF reports with graphs and task lists
organized by country, tech block (label), and week.
"""

import io
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle, KeepTogether
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from meeting_notes import get_note_for_date, get_all_notes


def create_matplotlib_graph(graph_data_dict, max_effort_func):
    """
    Create a matplotlib bar chart with horizontal max capacity lines.
    
    Args:
        graph_data_dict: Dictionary containing graph data (labels, values, max_capacity, etc.)
        max_effort_func: Function to get max effort for a label
    
    Returns:
        BytesIO buffer containing PNG image
    """
    data = graph_data_dict['data']
    title = graph_data_dict['title']
    
    labels = data['labels']
    values = data['values']
    max_capacity = data['max_capacity']
    within_capacity = data['within_capacity']
    over_capacity = data['over_capacity']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up x positions
    x_pos = np.arange(len(labels))
    
    # Color palette matching the original
    color_max = '#F5F5F5'  # Very light gray
    color_current = '#4A90E2'  # Modern blue
    color_overload = '#E74C3C'  # Modern red
    
    # Draw max capacity bars (background)
    ax.bar(x_pos, max_capacity, color=color_max, alpha=0.3, label='Max Capacity', edgecolor='gray', linewidth=1)
    
    # Draw within capacity bars
    ax.bar(x_pos, within_capacity, color=color_current, label='Current Effort', edgecolor='#2E6DA4', linewidth=1.2)
    
    # Draw overload bars (stacked on top)
    if any(oc > 0 for oc in over_capacity):
        ax.bar(x_pos, over_capacity, bottom=within_capacity, color=color_overload, 
               label='Overload', edgecolor='#C0392B', linewidth=1.2)
    
    # Add horizontal lines for max capacity (horizontal bars)
    for i, max_cap in enumerate(max_capacity):
        ax.hlines(max_cap, i - 0.4, i + 0.4, colors='gray', linestyles='dashed', 
                 linewidth=2, alpha=0.7, zorder=3)
    
    # Customize the plot
    ax.set_xlabel('Label', fontsize=11, color='#34495E')
    ax.set_ylabel('Effort (hours)', fontsize=11, color='#34495E')
    ax.set_title(title, fontsize=14, color='#2C3E50', fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save to BytesIO buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    
    return buffer


def generate_pdf_report(plan, all_ranges, plot_effort_graphs_func, get_active_task_by_label_func, 
                        max_effort_func, use_filters=True, 
                        selected_ranges=None, selected_labels=None, 
                        selected_countries=None, selected_projects=None,
                        analysis_date=None, max_capacity_values=None):
    """
    Generate a PDF report with graphs and task lists organized by country, tech block (label), and week.

    Args:
        plan: Plan object
        all_ranges: Dictionary of all date ranges
        plot_effort_graphs_func: Function to generate effort graphs
        get_active_task_by_label_func: Function to get active tasks by label
        max_effort_func: Function to get max effort for a label
        use_filters: If True, use current filter selections; if False, export all data
        selected_ranges: List of selected week ranges (if use_filters=True)
        selected_labels: List of selected labels (if use_filters=True)
        selected_countries: List of selected countries (if use_filters=True)
        selected_projects: List of selected projects (if use_filters=True)
        analysis_date: Analysis date for the report
        max_capacity_values: Dictionary of max capacity values per label (hours/day)
    """
    # Create a BytesIO buffer for the PDF
    buffer = io.BytesIO()
    
    # Create PDF document in landscape orientation
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), 
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,  # Max font size 14
        textColor=colors.HexColor('#34495E'),
        spaceAfter=12,
        spaceBefore=12
    )
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,  # Max font size 14, using 12
        textColor=colors.HexColor('#7F8C8D'),
        spaceAfter=8,
        spaceBefore=8
    )
    # Week title style - more visible with larger font, bold, and background
    week_title_style = ParagraphStyle(
        'WeekTitle',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#FFFFFF'),
        backColor=colors.HexColor('#2C3E50'),
        spaceAfter=12,
        spaceBefore=12,
        leftIndent=10,
        rightIndent=10
    )
    
    # Title
    title_text = f"Engineering Plan Review - 3 Week Report"
    if analysis_date:
        title_text += f"<br/><font size=12>Analysis Date: {analysis_date.strftime('%Y-%m-%d')}</font>"
    elements.append(Paragraph(title_text, title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Filter information
    if use_filters:
        filter_text = "Export Mode: <b>Filtered Data</b><br/>"
        if selected_ranges:
            filter_text += f"Week Ranges: {', '.join(selected_ranges)}<br/>"
        if selected_labels:
            filter_text += f"Labels: {', '.join(selected_labels)}<br/>"
        if selected_countries:
            filter_text += f"Countries: {', '.join(selected_countries)}<br/>"
        if selected_projects:
            filter_text += f"Projects: {', '.join(selected_projects)}<br/>"
    else:
        filter_text = "Export Mode: <b>All Data</b>"
    
    elements.append(Paragraph(filter_text, styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Section 1: Meeting Notes (First Section)
    elements.append(Paragraph("1. Meeting Notes", heading_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Get note for analysis date
    if analysis_date:
        analysis_note = get_note_for_date(datetime.combine(analysis_date, datetime.min.time()))
        if analysis_note:
            elements.append(Paragraph(f"<b>Notes for {analysis_date.strftime('%Y-%m-%d')}:</b>", subheading_style))
            elements.append(Spacer(1, 0.1*inch))
            # Split note into paragraphs for better formatting
            note_lines = analysis_note.split('\n')
            for line in note_lines:
                if line.strip():
                    # Escape HTML special characters and preserve line breaks
                    escaped_line = line.strip().replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    elements.append(Paragraph(escaped_line, styles['Normal']))
            elements.append(Spacer(1, 0.2*inch))
        else:
            elements.append(Paragraph("No meeting notes available for the analysis date.", styles['Normal']))
            elements.append(Spacer(1, 0.2*inch))
    else:
        elements.append(Paragraph("No analysis date provided.", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
    
    elements.append(PageBreak())
    
    # Determine which ranges and filters to use
    if use_filters and selected_ranges:
        ranges_to_export = {title: all_ranges[title] for title in selected_ranges if title in all_ranges}
        labels_to_export = selected_labels if selected_labels else None
        countries_to_export = selected_countries if selected_countries else ['Chile', 'Brasil', 'Mexico', 'Colombia']
        projects_to_export = selected_projects if selected_projects else None
    else:
        ranges_to_export = all_ranges
        labels_to_export = None
        countries_to_export = ['Chile', 'Brasil', 'Mexico', 'Colombia']
        projects_to_export = None
    
    # Section 2: Effort by Label - Organized by Week
    elements.append(Paragraph("2. Effort by Label", heading_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Generate graphs for all weeks
    graph_data = plot_effort_graphs_func(plan, ranges_to_export, label_filter=labels_to_export)
    
    # Get active tasks for each week
    all_active_tasks_by_week = []
    range_titles = []
    for title, (start, end) in ranges_to_export.items():
        active_tasks = get_active_task_by_label_func(plan.get_all_tasks(), start, end)
        all_active_tasks_by_week.append((title, active_tasks, (start, end)))
        range_titles.append(title)
    
    # Organize by week: Graph -> Table -> Tasks
    for week_idx, (week_title, weekly_tasks, (start, end)) in enumerate(all_active_tasks_by_week):
        # Find corresponding graph for this week
        graph_info = None
        for g_info in graph_data:
            if g_info['title'] == week_title:
                graph_info = g_info
                break
        
        # Page break before each week (except the first one)
        if week_idx > 0:
            elements.append(PageBreak())
        
        # Week header
        elements.append(Paragraph(week_title, subheading_style))
        elements.append(Spacer(1, 0.1*inch))
        
        # 1. Graph for this week
        if graph_info:
            try:
                # Create matplotlib graph
                img_buffer = create_matplotlib_graph(graph_info, max_effort_func)
                img = Image(img_buffer, width=7*inch, height=4.3*inch)
                elements.append(img)
                elements.append(PageBreak())  # Page break after graph, before table
            except Exception as e:
                elements.append(Paragraph(
                    f"<i>Graph image generation failed.</i><br/>"
                    f"Error: {str(e)}", 
                    styles['Normal']
                ))
                elements.append(PageBreak())
        
        # 2. Table for this week
        if graph_info:
            try:
                data = graph_info['data']
                table_data = [['Label', 'Total Effort (h)', 'Max Capacity (h)', 'Within Capacity (h)', 'Overload (h)']]
                
                for i, label in enumerate(data['labels']):
                    table_data.append([
                        label,
                        f"{data['values'][i]:.1f}",
                        f"{data['max_capacity'][i]:.1f}",
                        f"{data['within_capacity'][i]:.1f}",
                        f"{data['over_capacity'][i]:.1f}"
                    ])
                
                table = Table(table_data, colWidths=[3*inch, 1.6*inch, 1.6*inch, 1.6*inch, 1.6*inch], repeatRows=1)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A90E2')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),  # Max font 14, using 12 for header
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                    ('TOPPADDING', (0, 0), (-1, 0), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 11),  # Max font 14, using 11 for body
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                elements.append(KeepTogether(table))
                elements.append(PageBreak())  # Page break after table, before tasks
            except Exception as e:
                elements.append(Paragraph(f"Error generating table: {str(e)}", styles['Normal']))
                elements.append(PageBreak())
        
        # 3. Tasks for this week (organized by country)
        elements.append(Paragraph("Active Tasks by Country and Tech Block", subheading_style))
        elements.append(Spacer(1, 0.1*inch))
    
        # Get all countries from tasks for this week
        week_countries = set()
        for label, tasks in weekly_tasks.items():
            for task in tasks:
                if not pd.isna(task.country):
                    week_countries.add(task.country)
        
        # Filter countries if needed
        if countries_to_export:
            week_countries = [c for c in sorted(week_countries) if c in countries_to_export]
        else:
            week_countries = sorted(week_countries)
        
        # Organize by country -> label
        for country in week_countries:
            elements.append(Paragraph(f"<b>{country}</b>", styles['Heading4']))
            
            # Get all labels for this country
            country_labels = set()
            for label, tasks in weekly_tasks.items():
                # Filter by label if needed
                if labels_to_export and label not in labels_to_export:
                    continue
                for task in tasks:
                    if task.country == country:
                        if not pd.isna(task.label):
                            country_labels.add(task.label)
            
            country_labels = sorted(country_labels)
            
            for label in country_labels:
                elements.append(Paragraph(f"<i>Tech Block (Label): {label}</i>", styles['Normal']))
                
                # Collect tasks for this country and label
                task_list = []
                # Get tasks for this label from weekly_tasks (which is organized by label)
                if label in weekly_tasks:
                    # Filter by label if needed
                    if labels_to_export and label not in labels_to_export:
                        continue
                    for task in weekly_tasks[label]:
                        if task.country == country and task.label == label:
                            # Filter by project if needed
                            if projects_to_export:
                                if pd.isna(task.project) or task.project not in projects_to_export:
                                    continue
                            if not task.is_summary:
                                task_list.append(task)
                
                if task_list:
                    # Create table for tasks - removed "Assigned To" column
                    task_table_data = [['Task Name', 'Project', 'Start Date', 'Due Date', 'Effort (h)', 'Status']]
                    overdue_rows = []  # Track which rows are overdue
                    
                    for idx, task in enumerate(task_list):
                        is_overdue = task.is_overdue()
                        if is_overdue:
                            overdue_rows.append(idx + 1)  # +1 because row 0 is header
                        
                        status = "🔴 Overdue" if is_overdue else "⚪ Active"
                        start_str = task.start_date.strftime('%Y-%m-%d') if task.start_date else 'N/A'
                        due_str = task.due_date.strftime('%Y-%m-%d') if task.due_date else 'N/A'
                        effort_str = f"{task.effort:.1f}" if task.effort else 'N/A'
                        project_str = str(task.project) if not pd.isna(task.project) else 'N/A'
                        
                        # Don't truncate - let table handle wrapping
                        task_table_data.append([
                            task.name,
                            project_str,
                            start_str,
                            due_str,
                            effort_str,
                            status
                        ])
                    
                    if len(task_table_data) > 1:  # Has data rows
                        # Column widths adjusted for landscape layout (total ~9.5 inches)
                        # Removed "Assigned To" column, redistributed space
                        task_table = Table(task_table_data, colWidths=[
                            4*inch,    # Task Name - increased for landscape
                            2.5*inch,  # Project - increased
                            1.2*inch,  # Start Date
                            1.2*inch,  # Due Date
                            0.8*inch,  # Effort (h)
                            0.8*inch   # Status
                        ])
                        
                        # Build table style
                        table_style = [
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495E')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('ALIGN', (2, 0), (3, -1), 'CENTER'),  # Center dates
                            ('ALIGN', (4, 0), (4, -1), 'CENTER'),   # Center effort
                            ('ALIGN', (5, 0), (5, -1), 'CENTER'),   # Center status
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 8),  # Reduced header font
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                            ('TOPPADDING', (0, 0), (-1, 0), 10),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                            ('FONTSIZE', (0, 1), (-1, -1), 7),  # Reduced body font for better fit
                            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')]),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),  # Top align for multi-line content
                            ('WORDWRAP', (0, 0), (-1, -1), True),  # Enable word wrapping
                        ]
                        
                        # Add red text color for overdue tasks (all columns in overdue rows)
                        for row in overdue_rows:
                            table_style.append(('TEXTCOLOR', (0, row), (-1, row), colors.HexColor('#E74C3C')))  # Red color
                        
                        task_table.setStyle(TableStyle(table_style))
                        elements.append(task_table)
                        elements.append(Spacer(1, 0.15*inch))
                else:
                    elements.append(Paragraph("No tasks found.", styles['Normal']))
                    elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Spacer(1, 0.3*inch))
    
    # Section 3: Max Capacity Summary
    elements.append(PageBreak())
    elements.append(Paragraph("3. Max Capacity Summary", heading_style))
    elements.append(Spacer(1, 0.1*inch))
    
    if max_capacity_values and len(max_capacity_values) > 0:
        # Create table with max capacity values
        capacity_table_data = [['Label', 'Max Capacity (hours/day)']]
        
        # Sort labels alphabetically
        sorted_labels = sorted(max_capacity_values.keys())
        
        for label in sorted_labels:
            capacity_value = max_capacity_values[label]
            capacity_table_data.append([
                label,
                f"{capacity_value:.1f}"
            ])
        
        # Create table
        capacity_table = Table(capacity_table_data, colWidths=[4*inch, 2*inch], repeatRows=1)
        capacity_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495E')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),  # Center capacity values
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')]),
        ]))
        elements.append(KeepTogether(capacity_table))
        elements.append(Spacer(1, 0.2*inch))
        
        # Add explanatory text
        elements.append(Paragraph(
            "<i>Note: Max capacity values represent the maximum daily effort capacity (in hours) "
            "for each label. These values are used in the effort graphs to calculate capacity limits and overload.</i>",
            styles['Normal']
        ))
    else:
        elements.append(Paragraph(
            "No max capacity values configured. Using default values from the system.",
            styles['Normal']
        ))
    """
    # Build PDF
    doc.build(elements)
    
    # Get PDF bytes
    buffer.seek(0)
    return buffer.getvalue()
