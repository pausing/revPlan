import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from collections import defaultdict
from plan import Plan
from plannerReview import build_plan_from_outline
import os
import shutil

# Page configuration
st.set_page_config(page_title="Engineering Plan Review", layout="wide")

# Helper functions (extracted from Plan.createReport)
def get_daily_effort_by_label(tasks, from_date, to_date, seen=None):
    effort_by_label = defaultdict(float)
    
    if seen is None:
        seen = set()
    
    for task in tasks:
        if id(task) in seen:
            continue
        seen.add(id(task))
        
        if task.is_summary:
            subtasks_effort = get_daily_effort_by_label(task.subtasks, from_date, to_date, seen)
            for label, value in subtasks_effort.items():
                effort_by_label[label] += value
        else:
            if task.daily_effort and task.due_date and active_during_analysis_period(task, from_date, to_date):
                label = task.label or 'No Label'
                if pd.isna(label):
                    label = 'No Label'
                subtotalEffort = float(task.daily_effort) * ((to_date - from_date).days + 1)
                if pd.isna(subtotalEffort):
                    subtotalEffort = 0
                effort_by_label[label] += subtotalEffort
    return effort_by_label

def active_during_analysis_period(task, from_date, to_date):
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

def maxEffort(label):
    max_Eff = {
        'BESS': 16,  # 2 people
        'Civil Engineering': 16,  # 2 people
        'Eng Coordinators': 16,  # 3 people
        'Grid & HV': (16+28),  # 2 people # 3 people, 1 half time
        'PV': 28,  # 3 people, 1 half time
        'Yield Assessment ': 16,  # 2 people
    }
    try:
        return max_Eff[label]
    except:
        return 0

def define3WeekRanges(iniDate):
    today = iniDate
    
    start_of_week = today - timedelta(days=today.weekday())
    str_start_of_week = start_of_week.strftime('%Y-%m-%d')
    
    end_of_week = start_of_week + timedelta(days=4)
    str_end_of_week = end_of_week.strftime('%Y-%m-%d')
    
    start_next_week = start_of_week + timedelta(days=7)
    str_start_next_week = start_next_week.strftime('%Y-%m-%d')
    
    end_of_next_week = start_next_week + timedelta(days=4)
    str_end_next_week = end_of_next_week.strftime('%Y-%m-%d')
    
    start_in_3 = start_of_week + timedelta(days=14)
    str_start_in_3 = start_in_3.strftime('%Y-%m-%d')
    
    end_of_in_3_week = start_in_3 + timedelta(days=4)
    str_end_of_in_3_week = end_of_in_3_week.strftime('%Y-%m-%d')
    
    ranges = {
        'This week: {} to {}'.format(str_start_of_week, str_end_of_week): (start_of_week, start_of_week + timedelta(days=4)),
        'Next week: {} to {}'.format(str_start_next_week, str_end_next_week): (start_next_week, start_next_week + timedelta(days=4)),
        'In 3+ week: {} to {}'.format(str_start_in_3, str_end_of_in_3_week): (start_in_3, start_in_3 + timedelta(days=4)),
    }
    
    return ranges

def plot_effort_graphs(plan, rangesDates):
    """Create interactive effort graphs using Plotly"""
    tasks = plan.get_all_tasks()
    
    # Modern color palette
    colors = {
        'max_effort': '#F5F5F5',  # Very light gray
        'current_effort': '#4A90E2',  # Modern blue
        'overload': '#E74C3C',  # Modern red
    }
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[title for title in rangesDates.keys()],
        shared_yaxes=True,
        horizontal_spacing=0.08
    )
    
    ranges = rangesDates
    
    for i, (title, (start, end)) in enumerate(ranges.items()):
        effort_data = get_daily_effort_by_label(tasks, start, end)
        numDays = (end - start).days + 1
        
        labels = list(effort_data.keys())
        values = list(effort_data.values())
        max_capacity = [maxEffort(l) * numDays for l in labels]
        within_capacity = [min(c, m) * numDays for c, m in zip(values, max_capacity)]
        over_capacity = [max(0, c - m) * numDays for c, m in zip(values, max_capacity)]
        
        col_idx = i + 1
        
        # Calculate actual effort values (within + overload)
        actual_effort = [wc + oc for wc, oc in zip(within_capacity, over_capacity)]
        
        # Max effort background bars (shows capacity limit)
        fig.add_trace(
            go.Bar(
                x=labels,
                y=max_capacity,
                name='Max Capacity',
                marker_color=colors['max_effort'],
                marker_line_color='#E0E0E0',
                marker_line_width=1.2,
                opacity=0.6,
                showlegend=(i == 0),  # Only show legend for first subplot
                legendgroup='max',
                hovertemplate='<b>%{x}</b><br>Max Capacity: %{y:.1f} hours<extra></extra>'
            ),
            row=1, col=col_idx
        )
        
        # Current effort bars (within capacity portion)
        fig.add_trace(
            go.Bar(
                x=labels,
                y=within_capacity,
                name='Current Effort',
                marker_color=colors['current_effort'],
                marker_line_color='#2E6DA4',
                marker_line_width=1.2,
                opacity=0.9,
                showlegend=(i == 0),
                legendgroup='current',
                hovertemplate='<b>%{x}</b><br>Current Effort: %{y:.1f} hours<extra></extra>'
            ),
            row=1, col=col_idx
        )
        
        # Overload bars (stacked on top of current effort)
        if any(oc > 0 for oc in over_capacity):
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=over_capacity,
                    name='Overload',
                    marker_color=colors['overload'],
                    marker_line_color='#C0392B',
                    marker_line_width=1.2,
                    opacity=0.9,
                    showlegend=(i == 0),
                    legendgroup='overload',
                    hovertemplate='<b>%{x}</b><br>Overload: %{y:.1f} hours<extra></extra>'
                ),
                row=1, col=col_idx
            )
    
    # Update layout with modern styling
    # Use overlay mode so max capacity shows as background, effort bars overlay on top
    fig.update_layout(
        title={
            'text': 'Effort by Label',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2C3E50'}
        },
        yaxis_title='Effort (hours)',
        template='plotly_white',
        height=500,
        hovermode='x unified',
        barmode='overlay',  # Overlay bars so max capacity is background
        margin=dict(l=50, r=50, t=100, b=80),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=10)
        ),
        font=dict(family='Arial, sans-serif', size=11, color='#34495E')
    )
    
    # Manually stack current effort and overload by setting base parameter
    # Process each subplot's traces
    trace_idx = 0
    for i, (title, (start, end)) in enumerate(ranges.items()):
        effort_data = get_daily_effort_by_label(tasks, start, end)
        numDays = (end - start).days + 1
        labels = list(effort_data.keys())
        values = list(effort_data.values())
        max_capacity = [maxEffort(l) * numDays for l in labels]
        within_capacity = [min(c, m) * numDays for c, m in zip(values, max_capacity)]
        over_capacity = [max(0, c - m) * numDays for c, m in zip(values, max_capacity)]
        
        # Max capacity trace (trace_idx) - keep as is (background, already at base=0)
        # Current effort trace (trace_idx + 1) - starts from 0
        if trace_idx + 1 < len(fig.data):
            fig.data[trace_idx + 1].base = [0] * len(labels)
        
        # Overload trace (trace_idx + 2) - stacks on current effort (if it exists)
        has_overload = any(oc > 0 for oc in over_capacity)
        if has_overload and trace_idx + 2 < len(fig.data):
            fig.data[trace_idx + 2].base = within_capacity
        
        # Move to next subplot's traces (3 traces per subplot: max, current, [overload if exists])
        trace_idx += (3 if has_overload else 2)
    
    # Update x-axes
    for i in range(1, 4):
        fig.update_xaxes(
            tickangle=45,
            row=1, col=i,
            title_text='Label' if i == 2 else '',
            title_font=dict(size=11, color='#34495E')
        )
    
    # Update y-axis (shared)
    fig.update_yaxes(
        title_text='Effort (hours)',
        row=1, col=1,
        title_font=dict(size=11, color='#34495E')
    )
    
    return fig

def find_incomplete_tasks(tasks, seen=None):
    """Returns a list of unique tasks missing any of: project, effort, country, or label."""
    if seen is None:
        seen = set()
    
    incomplete = []
    
    for task in tasks:
        if id(task) in seen or task.is_summary == True or task.completed:
            continue
        
        seen.add(id(task))
        
        if (
            pd.isna(task.project)
            or pd.isna(task.effort)
            or pd.isna(task.country)
            or pd.isna(task.label)
        ):
            incomplete.append(task)
        
        if task.subtasks:
            incomplete += find_incomplete_tasks(task.subtasks, seen)
    
    return incomplete

def get_active_task_by_label(tasks, from_date, to_date, seen=None):
    activeTask_by_label = defaultdict(list)
    
    if seen is None:
        seen = set()
    
    for task in tasks:
        if id(task) in seen:
            continue
        seen.add(id(task))
        
        if task.is_summary:
            sub_active = get_active_task_by_label(task.subtasks, from_date, to_date, seen)
            for label, sublist in sub_active.items():
                activeTask_by_label[label].extend(sublist)
        else:
            if active_during_analysis_period(task, from_date, to_date):
                label = task.label or 'No Label'
                if pd.isna(label):
                    label = 'No Label'
                activeTask_by_label[label].extend([task])
    
    return activeTask_by_label

def preprocess_excel_file(file_path, data_folder='00_Data'):
    """
    Preprocess Excel file by extracting date from column B, line 7
    and renaming to AG_eng_planner_[date].xlsx
    """
    try:
        # Read the date from column B (index 1), row 7 (index 6 in 0-based)
        # Read first 10 rows to ensure we get line 7
        df = pd.read_excel(file_path, header=None, nrows=10)
        
        # Check if we have enough rows
        if len(df) < 7:
            return None, "File does not have enough rows. Line 7 not found."
        
        # Get value from column B (index 1), line 7 (index 6)
        date_value = df.iloc[6, 1]  # Row 7 (index 6), Column B (index 1)
        
        # Convert date to string format
        if pd.isna(date_value):
            return None, "Date not found in column B, line 7"
        
        # Handle different date formats
        date_str = None
        if isinstance(date_value, datetime):
            date_str = date_value.strftime('%Y%m%d')
        elif isinstance(date_value, pd.Timestamp):
            date_str = date_value.strftime('%Y%m%d')
        else:
            # Try to parse as date string
            try:
                date_obj = pd.to_datetime(str(date_value))
                date_str = date_obj.strftime('%Y%m%d')
            except:
                return None, f"Could not parse date from column B, line 7: {date_value}"
        
        if not date_str:
            return None, f"Could not extract date from: {date_value}"
        
        # Create new filename
        new_filename = f"AG_eng_planner_{date_str}.xlsx"
        new_file_path = os.path.join(data_folder, new_filename)
        
        # If the file is already named correctly, no need to rename
        if os.path.basename(file_path) == new_filename:
            return file_path, f"✅ File already has correct name: {new_filename}"
        
        # If target file already exists, remove it first
        if os.path.exists(new_file_path):
            os.remove(new_file_path)
        
        # Rename the file (move it)
        os.rename(file_path, new_file_path)
        
        return new_file_path, f"✅ File renamed to: {new_filename}"
    
    except Exception as e:
        return None, f"Error preprocessing file: {str(e)}"

# Main app
def main():
    st.title("📊 Engineering Plan Review - 3 Week Report")
    
    # Sidebar for data loading
    st.sidebar.header("Data Source")
    
    data_folder = '00_Data'
    
    # Ensure data folder exists
    os.makedirs(data_folder, exist_ok=True)
    
    # Preprocessing section
    st.sidebar.subheader("Preprocess Files")
    
    # Get list of Excel files in 00_Data folder for preprocessing
    files_to_preprocess = []
    if os.path.exists(data_folder):
        files_to_preprocess = [f for f in os.listdir(data_folder) 
                              if f.endswith(('.xlsx', '.xls'))]
        files_to_preprocess.sort()  # Sort alphabetically
    
    if files_to_preprocess:
        file_to_preprocess = st.sidebar.selectbox(
            "Select file to preprocess:",
            files_to_preprocess,
            help="Select a file from 00_Data folder to rename based on date in column B, line 7"
        )
        
        if st.sidebar.button("Preprocess File"):
            file_path = os.path.join(data_folder, file_to_preprocess)
            new_path, message = preprocess_excel_file(file_path, data_folder)
            
            if new_path:
                st.sidebar.success(message)
                st.rerun()  # Refresh to show updated file list
            else:
                st.sidebar.error(message)
    else:
        st.sidebar.info(f"No Excel files found in {data_folder} folder.")
    
    st.sidebar.divider()
    
    # File selection from 00_Data folder
    st.sidebar.subheader("Load Plan")
    
    # Get list of Excel files in 00_Data folder
    excel_files = []
    if os.path.exists(data_folder):
        excel_files = [f for f in os.listdir(data_folder) 
                      if f.endswith(('.xlsx', '.xls'))]
        excel_files.sort(reverse=True)  # Most recent first
    
    plan = None
    
    if excel_files:
        selected_file = st.sidebar.selectbox(
            "Select Excel file:",
            excel_files,
            help="Select a file from the 00_Data folder"
        )
        
        if selected_file:
            excel_file = os.path.join(data_folder, selected_file)
            
            if st.sidebar.button("Load from Excel"):
                try:
                    # Extract filename without extension for plan name
                    plan_name = os.path.splitext(os.path.basename(excel_file))[0]
                    plan = Plan(plan_name)
                    df = pd.read_excel(excel_file, skiprows=8)
                    build_plan_from_outline(df, plan)
                    st.sidebar.success(f"✅ Loaded plan: {plan.name}")
                except Exception as e:
                    st.sidebar.error(f"Error loading Excel: {e}")
            else:
                # Auto-load if file exists (for convenience)
                try:
                    # Extract filename without extension for plan name
                    plan_name = os.path.splitext(os.path.basename(excel_file))[0]
                    plan = Plan(plan_name)
                    df = pd.read_excel(excel_file, skiprows=8)
                    build_plan_from_outline(df, plan)
                except:
                    pass
    else:
        st.sidebar.info(f"No Excel files found in {data_folder} folder. Please upload and preprocess a file first.")
    
    if plan is None:
        st.info("👈 Please load a plan from the sidebar to get started.")
        return
    
    # Analysis date selector
    st.sidebar.header("Analysis Settings")
    analysis_date = st.sidebar.date_input(
        "Analysis Date",
        value=datetime.today().date()
    )
    
    # Convert to datetime
    start_analysis = datetime.combine(analysis_date, datetime.min.time())
    
    # Get date ranges
    ranges = define3WeekRanges(start_analysis)
    
    # Get active tasks for each week (needed to populate project filter)
    activeTasks = []
    for i, (title, (start, end)) in enumerate(ranges.items()):
        activeTasks.append(get_active_task_by_label(plan.get_all_tasks(), start, end))
    
    # Country filter for Active Tasks
    all_countries = ['Chile', 'Brasil', 'Mexico', 'Colombia']
    selected_countries = st.sidebar.multiselect(
        "Filter by Country (Active Tasks)",
        options=all_countries,
        default=all_countries,
        help="Select countries to display in Active Tasks section"
    )
    
    # Extract unique projects from active tasks filtered by selected countries
    all_projects = set()
    countries_to_filter = selected_countries if selected_countries else []
    
    if countries_to_filter:
        for weeklyTasks in activeTasks:
            for label, tasks in weeklyTasks.items():
                for task in tasks:
                    if task.country in countries_to_filter and not pd.isna(task.project):
                        all_projects.add(task.project)
    
    # Project filter for Active Tasks
    project_options = sorted(list(all_projects)) if all_projects else []
    selected_projects = st.sidebar.multiselect(
        "Filter by Project (Active Tasks)",
        options=project_options,
        default=project_options,
        help="Select projects to display in Active Tasks section (filtered by selected countries)"
    )
    
    # Main content
    st.header("1. Effort by Label")
    
    # Create and display graphs
    fig = plot_effort_graphs(plan, ranges)
    st.plotly_chart(fig, width='stretch')
    
    st.header("2. Active Tasks")
    
    # Show filter info
    filter_info = []
    if len(selected_countries) < len(all_countries):
        filter_info.append(f"Countries: {', '.join(selected_countries) if selected_countries else 'None'}")
    if len(selected_projects) < len(project_options):
        filter_info.append(f"Projects: {', '.join(selected_projects) if selected_projects else 'None'}")
    
    if filter_info:
        st.info(f"📌 Filters: {' | '.join(filter_info)}")
    
    # Use selected countries and projects for filtering
    countries_to_display = selected_countries if selected_countries else []
    projects_to_display = selected_projects if selected_projects else []
    
    # Display active tasks organized by week -> country -> label
    for i, weeklyTasks in enumerate(activeTasks):
        week_title = list(ranges.keys())[i]
        st.subheader(week_title)
        
        # If no countries selected, show message
        if not countries_to_display:
            st.warning("⚠️ Please select at least one country from the sidebar filter.")
            continue
        
        # If no projects selected, show message
        if not projects_to_display:
            st.warning("⚠️ Please select at least one project from the sidebar filter.")
            continue
        
        for c in countries_to_display:
            country_has_tasks = False
            for label, tasks in weeklyTasks.items():
                # Filter by both country and project
                tasksFiltered = [t for t in tasks 
                               if t.country == c 
                               and (not pd.isna(t.project) and t.project in projects_to_display)]
                if len(tasksFiltered) > 0:
                    country_has_tasks = True
                    break
            
            if country_has_tasks:
                st.markdown(f"**{c}**")
                
                for label, tasks in weeklyTasks.items():
                    # Filter by both country and project
                    tasksFiltered = [t for t in tasks 
                                   if t.country == c 
                                   and (not pd.isna(t.project) and t.project in projects_to_display)]
                    if len(tasksFiltered) > 0:
                        st.markdown(f"*Label: {label}*")
                        for t in tasksFiltered:
                            if t.is_overdue():
                                st.markdown(f":red[🔴 {str(t)}]")
                            else:
                                st.markdown(f"⚪ {str(t)}")
    
    st.header("3. Tasks with Missing Information")
    
    incompleteTasks = find_incomplete_tasks(plan.get_all_tasks())
    
    if len(incompleteTasks) == 0:
        st.success("✅ All tasks have complete information!")
    else:
        st.warning(f"⚠️ Found {len(incompleteTasks)} tasks with missing information:")
        
        for t in incompleteTasks:
            missing = []
            if pd.isna(t.project):
                missing.append("project")
            if pd.isna(t.effort):
                missing.append("effort")
            if pd.isna(t.country):
                missing.append("country")
            if pd.isna(t.label):
                missing.append("label")
            fields = ", ".join(missing)
            st.markdown(f"- **{t.name}**: Missing [{fields}]")

if __name__ == "__main__":
    main()

