from plan import Plan
import shutil
from task import Task
import os
import pandas as pd
import openpyxl as pyxl
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime, timedelta



def build_plan_from_outline(df,plan):
    df = df.sort_values(by='Outline number')  # Ensure hierarchy order

    outline_map = {}  # Map outline number -> Task

    for _, row in df.iterrows():
        outline = str(row['Outline number']).strip()
        name = row['Name']
        priority = row.get('Priority', 'Normal')
        assigned = row.get('Assigned To')
        country = row.get('Country')
        project = row.get('Project_')
        bucket = row.get('Bucket')
        label = row.get('Labels')
        start_date = pd.to_datetime(row['Start']) if pd.notnull(row['Start']) else None
        due_date = pd.to_datetime(row['Finish']) if pd.notnull(row['Finish']) else None
        effort = str(row.get('Effort')).split(' ')[0]
        complete = row.get('Completed')

        task = Task(
            name=name,
            priority=priority,
            assignedTo=assigned,
            country=country,
            project=project,
            bucket=bucket,
            label=label,
            start_date=start_date,
            due_date=due_date,
            effort=effort
        )

        # Determine if this is a summary task by checking if others are nested under it
        is_summary = any(o.startswith(outline + '.') for o in df['Outline number'] if isinstance(o, str))
        task.is_summary = is_summary

        outline_map[outline] = task

        # Find parent by removing the last segment (e.g., 1.2.3 -> 1.2)
        if '.' in outline:
            parent_outline = '.'.join(outline.split('.')[:-1])
            parent = outline_map.get(parent_outline)
            if parent:
                parent.add_subtask(task)
        else:
            # Top-level task
            plan.add_task(task)

    return plan

def get_effort_by_label(tasks,from_date,to_date):
    from collections import defaultdict
    effort_by_label = defaultdict(float)

    for task in tasks:
        if task.is_summary:
            subtasks_effort = get_effort_by_label(task.subtasks,from_date,to_date)
            for label, value in subtasks_effort.items():
                effort_by_label[label] += value
        else:
            if task.effort and task.due_date and from_date <= task.due_date < to_date:
                label = task.label or 'No Label'
                if pd.isna(label):
                    label = 'No Label'
                effort_by_label[label] += float(task.effort)
    return effort_by_label

def plot_effort_graphs(plan):
    tasks = plan.get_all_tasks()
    fig, axes = plt.subplots(1,3,figsize=(18,5),sharey = True)

    today = datetime.today()
    start_of_week = today - timedelta(days=today.weekday())
    start_next_week = start_of_week + timedelta(days=7)
    start_in_3 = start_of_week + timedelta(days=14)

    ranges = {
        'This week': (start_of_week,start_next_week),
        'Next week': (start_next_week,start_in_3),
        'In 3+ week': (start_in_3,start_in_3 + timedelta(days=7)),
    }

    for i, (title, (start, end)) in enumerate(ranges.items()):
        effort_data = get_effort_by_label(tasks,start,end)
        print(effort_data)

        labels = list(effort_data.keys())
        values = list(effort_data.values())

        axes[i].bar(labels,values,color='skyblue')
        axes[i].set_title(title)
        axes[i].set_ylabel('Effort (hours)')
        axes[i].tick_params(axis='x',rotation=45)
    
    plt.tight_layout()
    plt.show()



def main():
    bckupFile = 'test.json'
    exportPlanFile = 'Asset Growth Engineering 2025 (2).xlsx'

    shutil.copy(exportPlanFile,'test.xlsx')
    exFile = pyxl.open('test.xlsx')


    engPlan = Plan('engPlan')
    #if os.path.isfile(bckupFile):
        #engPlan.load_from_file(bckupFile)
    
    df = pd.read_excel(exportPlanFile,skiprows=8)
    
    build_plan_from_outline(df,engPlan)

    plot_effort_graphs(engPlan)
    
    print(engPlan)

    engPlan.save_to_file('testPlan.json')
    

    
    

    
    


if __name__ == '__main__':
    main()
