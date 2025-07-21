import json
from datetime import datetime
from task import Task
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from fpdf import FPDF

class Plan:
    def __init__(self, name):
        self.name = name
        self.created_at = datetime.now()
        self.tasks = []  # list of top-level Task objects

    def add_task(self, task):
        self.tasks.append(task)

    def get_all_tasks(self, include_subtasks=True):
        result = []

        def collect(task):
            result.append(task)
            if include_subtasks:
                for sub in task.subtasks:
                    collect(sub)

        for task in self.tasks:
            collect(task)

        return result

    def get_summary(self):
        all_tasks = self.get_all_tasks()
        total = len(all_tasks)
        done = sum(1 for t in all_tasks if t.completed)
        return {
            'total_tasks': total,
            'completed_tasks': done,
            'completion_rate': round(100 * done / total, 2) if total else 0
        }

    def to_dict(self):
        return {
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'tasks': [t.to_dict() for t in self.tasks]
        }

    @classmethod
    def from_dict(cls, data):
        plan = cls(data['name'])
        plan.created_at = datetime.fromisoformat(data['created_at'])
        plan.tasks = [Task.from_dict(t) for t in data['tasks']]
        return plan

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __str__(self):
        lines = [f"Plan: {self.name}", "-"*20]
        for task in self.tasks:
            lines.append(task.__str__())
        return '\n'.join(lines)
    
    def createReport(self,start_analysis,reportPath):
        def get_daily_effort_by_label(tasks,from_date,to_date,seen=None):
            from collections import defaultdict
            effort_by_label = defaultdict(float)

            if seen is None:
                seen = set()

            for task in tasks:
                if id(task) in seen:
                    continue
                seen.add(id(task))

                if task.is_summary:
                    subtasks_effort = get_daily_effort_by_label(task.subtasks,from_date,to_date,seen)
                    for label, value in subtasks_effort.items():
                        effort_by_label[label] += value
                else:
                    if task.daily_effort and task.due_date and active_during_analysis_period(task,from_date,to_date):
                        label = task.label or 'No Label'
                        if pd.isna(label):
                            label = 'No Label'
                        subtotalEffort = float(task.daily_effort)*((to_date-from_date).days + 1)
                        if pd.isna(subtotalEffort):
                            subtotalEffort = 0
                        #print(f'task {task.name} is active, for lable: {label} with daily effort of {task.daily_effort} and total effort {subtotalEffort} ')
                        effort_by_label[label] += subtotalEffort
            return effort_by_label

        def active_during_analysis_period(task,from_date,to_date):
            task_start = task.start_date
            task_finish = task.due_date
            active = False
            if task_start is None or task_finish is None:
                active = False
            else:
                if not task.completed:
                    if datetime.today() <= to_date and datetime.today() >= from_date:
                        #active = task_start <= to_date and task_finish >= from_date and not task.completed
                        active = task_start <= to_date
                    else:
                        active = task_start <= to_date and task_finish >= from_date and not task.completed
                        #active = task_start <= to_date
            return active

        def maxEffort(label):
            max_Eff = {
                'BESS':16, # 2 people
                'Civil Engineering':16, # 2 people
                'Eng Coordinators':16, # 3 people
                'Grid & HV':(16+28), # 2 people # 3 people, 1 half time
                'PV':28, # 3 people, 1 half time
                'Yield Assessment ':16, # 2 people
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
                'This week: {} to {}'.format(str_start_of_week,str_end_of_week): (start_of_week,start_of_week + timedelta(days=4)),
                'Next week: {} to {}'.format(str_start_next_week,str_end_next_week): (start_next_week,start_next_week + timedelta(days=4)),
                'In 3+ week: {} to {}'.format(str_start_in_3,str_end_of_in_3_week): (start_in_3,start_in_3 + timedelta(days=4)),
            }

            return ranges

        def plot_effort_graphs(plan, graphPath, rangesDates):
            tasks = plan.get_all_tasks()   
            fig, axes = plt.subplots(1,3,figsize=(18,5),sharey = True)
            ranges = rangesDates

            for i, (title, (start, end)) in enumerate(ranges.items()):
                effort_data = get_daily_effort_by_label(tasks,start,end)
                numDays = (end-start).days + 1
                #print(numDays)

                labels = list(effort_data.keys())
                #print(labels)
                values = list(effort_data.values())
                #print(values)
                max_capacity = [maxEffort(l)*numDays for l in labels]
                within_capacity = [min(c, m)*numDays for c, m in zip(values,max_capacity)]
                over_capacity = [max(0, c - m)*numDays for c, m in zip(values,max_capacity)]

                #axes[i].bar(labels,values,color='skyblue') # base capacity
                axes[i].bar(labels,max_capacity,color='lightgrey',label='max effort') # base capacity
                axes[i].bar(labels,within_capacity,color='blue',label='current effort') # load within capacity
                axes[i].bar(labels,over_capacity,bottom=within_capacity,color='red',label='overload') # excess load
                axes[i].set_title(title)
                axes[i].set_ylabel('Effort (hours)')
                axes[i].tick_params(axis='x',rotation=45)
    
            plt.tight_layout()
            plt.savefig(graphPath)
            plt.close()

        def find_incomplete_tasks(tasks, seen=None):
            """
            Returns a list of unique tasks missing any of: project, effort, country, or label.
            Uses pd.isna() to detect missing values including None, NaN, etc.
            """
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

        def get_active_task_by_label(tasks,from_date,to_date,seen=None):
            from collections import defaultdict
            activeTask_by_label = defaultdict(list)

            if seen is None:
                seen = set()

            for task in tasks:
                if id(task) in seen:
                    continue
                seen.add(id(task))

                if task.is_summary:
                    sub_active = get_active_task_by_label(task.subtasks,from_date,to_date,seen)
                    for label, sublist in sub_active.items():
                        activeTask_by_label[label].extend(sublist)
                else:
                    if active_during_analysis_period(task,from_date,to_date):
                        label = task.label or 'No Label'
                        if pd.isna(label):
                            label = 'No Label'
                        activeTask_by_label[label].extend([task])

            return activeTask_by_label
        

        
        # creat date ranges
        ranges = define3WeekRanges(start_analysis)
        
        # create graph
        graphPath = f'{start_analysis.strftime('%Y%m%d')}_graph_effort.png'
        plot_effort_graphs(self,graphPath,ranges)

        # list of active tasks
        activeTasks = []
        for i, (title,(start,end)) in enumerate(ranges.items()):
            activeTasks.append(get_active_task_by_label(self.get_all_tasks(),start,end))

        # list of incomplete tasks
        incompleteTasks = find_incomplete_tasks(self.get_all_tasks())

        # list of countries
        countries = ['Chile','Brasil','Mexico','Colombia']

        # create pdf
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True,margin=15)
        pdf.add_page()
        pdf.add_font('DejaVu','','DejaVuSans.ttf',uni=True)

        pdf.set_font('DejaVu','',16)
        pdf.cell(0,10,'3 week report',ln=True,align='C')
        pdf.ln(10)

        pdf.set_font('DejaVu', '', 12)
        pdf.cell(0, 10, '1. Effort by Label', ln=True)
        pdf.image(graphPath, w=180)
        pdf.ln(10)

        pdf.set_font('DejaVu', '', 12)
        pdf.cell(0, 10, '2. Active Tasks', ln=True)
        pdf.set_font('DejaVu', '', 10)

        for i,weeklyTasks in enumerate(activeTasks):
            pdf.ln(5)
            pdf.set_font('DejaVu', '', 10)
            pdf.cell(0, 8,list(ranges.keys())[i], ln=True)
            pdf.ln(5)

            for c in countries:
                pdf.ln(5)
                pdf.set_font('DejaVu', '', 10)
                pdf.cell(0, 8,c, ln=True)
                pdf.ln(5)

                for label,tasks in weeklyTasks.items():
                    tasksFiltered_countr = [t for t in tasks if t.country == c]
                    if len(tasksFiltered_countr) > 0:
                        pdf.set_font('DejaVu', '', 10)
                        pdf.cell(0, 8, f'Label: {label}', ln=True)
                        pdf.set_font('DejaVu', '', 6)
                        for t in tasks:
                            if t.country == c:
                                if not t.is_overdue():
                                    pdf.multi_cell(0, 6, str(t))
                                else:
                                    pdf.set_text_color(255,0,0)
                                    pdf.multi_cell(0, 6, str(t))
                                    pdf.set_text_color(0,0,0)

        pdf.ln(5)

        pdf.set_font('DejaVu', '', 12)
        pdf.cell(0, 10, '3. Tasks with Missing Information', ln=True)
        pdf.set_font('DejaVu', '', 6)
        for t in incompleteTasks:
            missing = []
            if pd.isna(t.project): missing.append("project")
            if pd.isna(t.effort): missing.append("effort")
            if pd.isna(t.country): missing.append("country")
            if pd.isna(t.label): missing.append("label")
            fields = ", ".join(missing)
            pdf.multi_cell(0, 6, f"- {t.name}: Missing [{fields}]")
        
        pdf.output(reportPath)

