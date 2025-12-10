import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from collections import defaultdict
from plan import Plan
from plannerReview import build_plan_from_outline
import os

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
        barmode='group',  # Group bars side by side for comparison
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

# Main app
def main():
    st.title("📊 Engineering Plan Review - 3 Week Report")
    
    # Sidebar for data loading
    st.sidebar.header("Data Source")
    
    load_option = st.sidebar.radio(
        "Load plan from:",
        ["JSON file", "Excel file"]
    )
    
    plan = None
    
    if load_option == "JSON file":
        json_file = st.sidebar.text_input("JSON file path", value="testPlan.json")
        if st.sidebar.button("Load from JSON"):
            try:
                if os.path.exists(json_file):
                    plan = Plan.load_from_file(json_file)
                    st.sidebar.success(f"✅ Loaded plan: {plan.name}")
                else:
                    st.sidebar.error(f"File not found: {json_file}")
            except Exception as e:
                st.sidebar.error(f"Error loading JSON: {e}")
        elif os.path.exists(json_file):
            # Auto-load if file exists (for convenience)
            try:
                plan = Plan.load_from_file(json_file)
            except:
                pass
    
    else:  # Excel file
        excel_file = st.sidebar.text_input("Excel file path", value="test.xlsx")
        if st.sidebar.button("Load from Excel"):
            try:
                if os.path.exists(excel_file):
                    plan = Plan('engPlan')
                    df = pd.read_excel(excel_file, skiprows=8)
                    build_plan_from_outline(df, plan)
                    st.sidebar.success(f"✅ Loaded plan from Excel: {plan.name}")
                else:
                    st.sidebar.error(f"File not found: {excel_file}")
            except Exception as e:
                st.sidebar.error(f"Error loading Excel: {e}")
        elif os.path.exists(excel_file):
            # Auto-load if file exists (for convenience)
            try:
                plan = Plan('engPlan')
                df = pd.read_excel(excel_file, skiprows=8)
                build_plan_from_outline(df, plan)
            except:
                pass
    
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
    
    # Main content
    st.header("1. Effort by Label")
    
    # Create and display graphs
    fig = plot_effort_graphs(plan, ranges)
    st.plotly_chart(fig, use_container_width=True)
    
    st.header("2. Active Tasks")
    
    # Get active tasks for each week
    activeTasks = []
    for i, (title, (start, end)) in enumerate(ranges.items()):
        activeTasks.append(get_active_task_by_label(plan.get_all_tasks(), start, end))
    
    # List of countries
    countries = ['Chile', 'Brasil', 'Mexico', 'Colombia']
    
    # Display active tasks organized by week -> country -> label
    for i, weeklyTasks in enumerate(activeTasks):
        week_title = list(ranges.keys())[i]
        st.subheader(week_title)
        
        for c in countries:
            country_has_tasks = False
            for label, tasks in weeklyTasks.items():
                tasksFiltered_countr = [t for t in tasks if t.country == c]
                if len(tasksFiltered_countr) > 0:
                    country_has_tasks = True
                    break
            
            if country_has_tasks:
                st.markdown(f"**{c}**")
                
                for label, tasks in weeklyTasks.items():
                    tasksFiltered_countr = [t for t in tasks if t.country == c]
                    if len(tasksFiltered_countr) > 0:
                        st.markdown(f"*Label: {label}*")
                        for t in tasksFiltered_countr:
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
    
    # Summary statistics
    st.sidebar.header("Summary")
    summary = plan.get_summary()
    st.sidebar.metric("Total Tasks", summary['total_tasks'])
    st.sidebar.metric("Completed Tasks", summary['completed_tasks'])
    st.sidebar.metric("Completion Rate", f"{summary['completion_rate']}%")
    st.sidebar.metric("Incomplete Tasks", len(incompleteTasks))

if __name__ == "__main__":
    main()

