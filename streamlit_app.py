import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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

def get_default_max_effort():
    """Returns the default max effort values"""
    return {
        'BESS': 16,  # 2 people
        'Civil Engineering': 16,  # 2 people
        'Eng Coordinators': 16,  # 3 people
        'Grid & HV': (16+28),  # 2 people # 3 people, 1 half time
        'PV': 28,  # 3 people, 1 half time
        'Yield Assessment ': 16,  # 2 people
    }

def maxEffort(label):
    """Get max effort for a label, using session state if available, otherwise defaults"""
    # Initialize session state with defaults if not present
    if 'max_capacity_values' not in st.session_state:
        st.session_state.max_capacity_values = get_default_max_effort()
    
    # Get from session state, fall back to default if not found
    max_Eff = st.session_state.max_capacity_values
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

def plot_effort_graphs(plan, rangesDates, label_filter=None):
    """Create interactive effort graphs using Plotly - returns list of dicts with figure and data
    
    Args:
        plan: Plan object
        rangesDates: Dictionary of date ranges
        label_filter: Optional list of labels to include (if None, includes all labels)
    """
    tasks = plan.get_all_tasks()
    
    # Modern color palette
    colors = {
        'max_effort': '#F5F5F5',  # Very light gray
        'current_effort': '#4A90E2',  # Modern blue
        'overload': '#E74C3C',  # Modern red
    }
    
    ranges = rangesDates
    graph_data = []
    
    for i, (title, (start, end)) in enumerate(ranges.items()):
        effort_data = get_daily_effort_by_label(tasks, start, end)
        
        # Filter by labels if label_filter is provided
        if label_filter:
            effort_data = {label: effort_data[label] for label in effort_data.keys() if label in label_filter}
        
        numDays = (end - start).days + 1
        
        labels = list(effort_data.keys())
        values = list(effort_data.values())
        max_capacity = [maxEffort(l) * numDays for l in labels]
        within_capacity = [min(c, m) for c, m in zip(values, max_capacity)]
        over_capacity = [max(0, c - m) for c, m in zip(values, max_capacity)]
        
        # Create independent figure for this time range
        fig = go.Figure()
        
        # Check if there's overload
        has_overload = any(oc > 0 for oc in over_capacity)
        
        # Current effort bars (within capacity portion) - always add first
        fig.add_trace(
            go.Bar(
                x=labels,
                y=within_capacity,
                name='Current Effort',
                marker_color=colors['current_effort'],
                marker_line_color='#2E6DA4',
                marker_line_width=1.2,
                opacity=0.9,
                showlegend=True,
                legendgroup='current',
                hovertemplate='<b>%{x}</b><br>Within Capacity: %{y:.1f} hours<extra></extra>'
            )
        )
        
        # Overload bars (stacked on top of current effort) - only if overload exists
        if has_overload:
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=over_capacity,
                    name='Overload',
                    marker_color=colors['overload'],
                    marker_line_color='#C0392B',
                    marker_line_width=1.2,
                    opacity=0.9,
                    showlegend=True,
                    legendgroup='overload',
                    hovertemplate='<b>%{x}</b><br>Overload: %{y:.1f} hours<extra></extra>'
                )
            )
        
        # Add max capacity as horizontal reference lines
        shapes = []
        for i, max_cap in enumerate(max_capacity):
            shapes.append(
                dict(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=i-0.4, x1=i+0.4,
                    y0=max_cap, y1=max_cap,
                    line=dict(color=colors['max_effort'], width=2, dash="dash"),
                    layer="below"
                )
            )
        
        # Add invisible trace for max capacity legend
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(color=colors['max_effort'], width=2, dash="dash"),
                name='Max Capacity',
                showlegend=True,
                legendgroup='max',
                hovertemplate='Max Capacity<extra></extra>'
            )
        )
        
        # Update layout with modern styling
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': '#2C3E50'}
            },
            yaxis_title='Effort (hours)',
            xaxis_title='Label',
            template='plotly_white',
            height=500,
            hovermode='x unified',
            barmode='stack',  # Stack mode for proper stacking of within_capacity + overload
            shapes=shapes,  # Add max capacity reference lines
            margin=dict(l=50, r=50, t=100, b=80),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                font=dict(size=10)
            ),
            font=dict(family='Arial, sans-serif', size=11, color='#34495E'),
            xaxis=dict(
                tickangle=45,
                title_font=dict(size=11, color='#34495E')
            ),
            yaxis=dict(
                title_font=dict(size=11, color='#34495E')
            )
        )
        
        # Store data for table display
        graph_data.append({
            'title': title,
            'figure': fig,
            'data': {
                'labels': labels,
                'values': values,
                'max_capacity': max_capacity,
                'within_capacity': within_capacity,
                'over_capacity': over_capacity,
                'has_overload': has_overload
            }
        })
    
    return graph_data

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
    all_ranges = define3WeekRanges(start_analysis)
    
    # Date range filter - filter by week ranges
    range_options = list(all_ranges.keys())
    selected_ranges = st.sidebar.multiselect(
        "Filter by Week Range",
        options=range_options,
        default=range_options,
        help="Select week ranges to display in graphs and active tasks"
    )
    
    # Filter ranges based on selection
    if selected_ranges:
        ranges = {title: all_ranges[title] for title in selected_ranges if title in all_ranges}
    else:
        ranges = {}
        st.sidebar.warning("⚠️ Please select at least one week range.")
    
    # Get active tasks for each week (needed to populate project filter)
    all_activeTasks = []
    for i, (title, (start, end)) in enumerate(all_ranges.items()):
        all_activeTasks.append(get_active_task_by_label(plan.get_all_tasks(), start, end))
    
    # Filter activeTasks based on selected ranges and create mapping
    filtered_activeTasks = []
    filtered_range_titles = []
    if selected_ranges:
        for i, title in enumerate(range_options):
            if title in selected_ranges:
                filtered_activeTasks.append(all_activeTasks[i])
                filtered_range_titles.append(title)
        activeTasks = filtered_activeTasks
    else:
        activeTasks = []
        filtered_range_titles = []
    
    # Extract all unique labels from active tasks and graphs
    all_labels = set()
    # Get labels from all active tasks (all ranges, not just filtered)
    for weeklyTasks in all_activeTasks:
        all_labels.update(weeklyTasks.keys())
    
    # Also get labels from all ranges to ensure we capture all possible labels
    for (start, end) in all_ranges.values():
        effort_data = get_daily_effort_by_label(plan.get_all_tasks(), start, end)
        all_labels.update(effort_data.keys())
    
    # Label filter
    label_options = sorted(list(all_labels)) if all_labels else []
    selected_labels = st.sidebar.multiselect(
        "Filter by Label",
        options=label_options,
        default=label_options,
        help="Select labels to display in graphs and active tasks"
    )
    
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
        for weeklyTasks in all_activeTasks:
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
    
    # Initialize session state for max capacity values
    if 'max_capacity_values' not in st.session_state:
        st.session_state.max_capacity_values = get_default_max_effort()
    
    # Main content with tabs
    tab1, tab2 = st.tabs(["📊 Reports", "⚙️ Max Capacity Settings"])
    
    with tab1:
        st.header("1. Effort by Label")
        
        # Check if ranges are selected
        if not ranges:
            st.warning("⚠️ Please select at least one week range from the sidebar filter to view graphs.")
        else:
            # Create and display graphs (filtered by selected ranges and labels)
            graph_data = plot_effort_graphs(plan, ranges, label_filter=selected_labels if selected_labels else None)
            for graph_info in graph_data:
                # Display the graph
                st.plotly_chart(graph_info['figure'], width='stretch')
                
                # Create expandable widget with table data
                with st.expander(f"📊 View data table: {graph_info['title']}"):
                    data = graph_info['data']
                    
                    # Create DataFrame for the table
                    df = pd.DataFrame({
                        'Label': data['labels'],
                        'Total Effort (hours)': [f"{v:.1f}" for v in data['values']],
                        'Max Capacity (hours)': [f"{m:.1f}" for m in data['max_capacity']],
                        'Within Capacity (hours)': [f"{w:.1f}" for w in data['within_capacity']],
                        'Overload (hours)': [f"{o:.1f}" for o in data['over_capacity']]
                    })
                    
                    # Display the table
                    st.dataframe(df, width='stretch', hide_index=True)
        
        st.header("2. Active Tasks")
        
        # Show filter info
        filter_info = []
        if len(selected_ranges) < len(range_options):
            filter_info.append(f"Week Ranges: {', '.join(selected_ranges) if selected_ranges else 'None'}")
        if selected_labels and len(selected_labels) < len(label_options):
            filter_info.append(f"Labels: {', '.join(selected_labels) if selected_labels else 'None'}")
        if len(selected_countries) < len(all_countries):
            filter_info.append(f"Countries: {', '.join(selected_countries) if selected_countries else 'None'}")
        if len(selected_projects) < len(project_options):
            filter_info.append(f"Projects: {', '.join(selected_projects) if selected_projects else 'None'}")
        
        if filter_info:
            st.info(f"📌 Filters: {' | '.join(filter_info)}")
        
        # Check if ranges are selected
        if not selected_ranges or not activeTasks:
            st.warning("⚠️ Please select at least one week range from the sidebar filter to view active tasks.")
        else:
            # Use selected countries, projects, and labels for filtering
            countries_to_display = selected_countries if selected_countries else []
            projects_to_display = selected_projects if selected_projects else []
            labels_to_display = selected_labels if selected_labels else []
            
            # Display active tasks organized by week -> country -> label
            for i, weeklyTasks in enumerate(activeTasks):
                if i < len(filtered_range_titles):
                    week_title = filtered_range_titles[i]
                    st.subheader(week_title)
                else:
                    continue
                
                # If no countries selected, show message
                if not countries_to_display:
                    st.warning("⚠️ Please select at least one country from the sidebar filter.")
                    continue
                
                # If no projects selected, show message
                if not projects_to_display:
                    st.warning("⚠️ Please select at least one project from the sidebar filter.")
                    continue
                
                # If labels are selected but none match, skip
                if labels_to_display:
                    # Check if any labels in weeklyTasks match selected labels
                    has_matching_labels = any(label in labels_to_display for label in weeklyTasks.keys())
                    if not has_matching_labels:
                        continue
                
                for c in countries_to_display:
                    country_has_tasks = False
                    for label, tasks in weeklyTasks.items():
                        # Filter by label if labels are selected
                        if labels_to_display and label not in labels_to_display:
                            continue
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
                            # Filter by label if labels are selected
                            if labels_to_display and label not in labels_to_display:
                                continue
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
    
    with tab2:
        st.header("⚙️ Max Capacity Settings")
        st.markdown("Configure the maximum daily effort capacity (in hours) for each label. These values are used in the effort graphs to calculate capacity limits and overload.")
        
        # Get all labels from the plan
        all_labels_in_plan = set()
        if plan:
            for task in plan.get_all_tasks():
                if hasattr(task, 'label') and task.label and not pd.isna(task.label):
                    all_labels_in_plan.add(task.label)
        
        # Get current values from session state
        current_values = st.session_state.max_capacity_values.copy()
        default_values = get_default_max_effort()
        
        # Combine all labels: defaults, current values, and labels from plan
        all_labels = sorted(list(set(
            list(default_values.keys()) + 
            list(current_values.keys()) + 
            list(all_labels_in_plan)
        )))
        
        # Add new label section
        with st.expander("➕ Add New Label", expanded=False):
            col_new1, col_new2 = st.columns([3, 1])
            with col_new1:
                new_label = st.text_input("Label Name", placeholder="Enter new label name")
            with col_new2:
                new_capacity = st.number_input("Capacity (hours/day)", min_value=0.0, max_value=200.0, value=16.0, step=1.0, key="new_capacity")
            
            if st.button("Add Label", key="add_label_btn"):
                if new_label and new_label.strip():
                    if new_label not in current_values:
                        current_values[new_label.strip()] = float(new_capacity)
                        st.session_state.max_capacity_values = current_values
                        st.success(f"✅ Added label '{new_label.strip()}' with capacity {new_capacity} hours/day")
                        st.rerun()
                    else:
                        st.warning(f"⚠️ Label '{new_label.strip()}' already exists. Edit it in the form below.")
                else:
                    st.error("❌ Please enter a label name.")
        
        # Create a form for editing values
        with st.form("max_capacity_form"):
            st.subheader("Edit Max Capacity Values")
            
            # Create input fields for each label
            updated_values = {}
            if all_labels:
                col1, col2 = st.columns(2)
                
                for i, label in enumerate(all_labels):
                    with col1 if i % 2 == 0 else col2:
                        current_val = current_values.get(label, default_values.get(label, 0))
                        updated_values[label] = st.number_input(
                            label=f"{label}",
                            min_value=0.0,
                            max_value=200.0,
                            value=float(current_val),
                            step=1.0,
                            help=f"Maximum daily effort capacity for {label} (hours)",
                            key=f"capacity_{label}"
                        )
            else:
                st.info("No labels found. Load a plan to see labels, or add new labels above.")
            
            # Form buttons (always show)
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                save_button = st.form_submit_button("💾 Save Changes", width='stretch')
            with col_btn2:
                reset_button = st.form_submit_button("🔄 Reset to Defaults", width='stretch')
            with col_btn3:
                clear_button = st.form_submit_button("🗑️ Clear All", width='stretch')
        
        # Handle form submissions
        if save_button:
            # Only save non-zero values
            filtered_values = {k: v for k, v in updated_values.items() if v > 0}
            st.session_state.max_capacity_values = filtered_values
            st.success("✅ Max capacity values saved! The graphs will update automatically.")
            st.rerun()
        
        if reset_button:
            st.session_state.max_capacity_values = get_default_max_effort()
            st.success("✅ Max capacity values reset to defaults!")
            st.rerun()
        
        if clear_button:
            st.session_state.max_capacity_values = {}
            st.success("✅ All max capacity values cleared!")
            st.rerun()
        
        # Display current values in a table
        st.subheader("Current Values")
        if st.session_state.max_capacity_values:
            values_df = pd.DataFrame({
                'Label': list(st.session_state.max_capacity_values.keys()),
                'Max Capacity (hours/day)': list(st.session_state.max_capacity_values.values())
            })
            values_df = values_df.sort_values('Label')
            st.dataframe(values_df, width='stretch', hide_index=True)
        else:
            st.info("No max capacity values configured. Using defaults.")
        
        # Show default values for reference
        with st.expander("📋 View Default Values"):
            default_df = pd.DataFrame({
                'Label': list(default_values.keys()),
                'Default Max Capacity (hours/day)': list(default_values.values())
            })
            default_df = default_df.sort_values('Label')
            st.dataframe(default_df, width='stretch', hide_index=True)
        
        # Show labels found in plan
        if all_labels_in_plan:
            with st.expander("📌 Labels Found in Current Plan"):
                labels_list = sorted(list(all_labels_in_plan))
                st.write(", ".join(labels_list))

if __name__ == "__main__":
    main()

