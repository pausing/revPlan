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
        completed = int(row.get('% complete')) == 1

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
            effort=effort,
            completed=completed
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

def main():
    bckupFile = 'test.json'
    exportPlanFile = '20250716_engPlan.xlsx'

    shutil.copy(exportPlanFile,'test.xlsx')

    engPlan = Plan('engPlan')
    #if os.path.isfile(bckupFile):
        #engPlan.load_from_file(bckupFile)
    
    df = pd.read_excel(exportPlanFile,skiprows=8)
    
    build_plan_from_outline(df,engPlan)

    #plot_effort_graphs(engPlan)
    d = datetime.today()
    engPlan.createReport(d,f'{d.strftime('%Y-%m-%d')}.pdf')
    
    #print(engPlan)
    #report_incomplete_tasks(engPlan.get_all_tasks())
    #print_active_task(engPlan)

    engPlan.save_to_file('testPlan.json')

if __name__ == '__main__':
    main()
