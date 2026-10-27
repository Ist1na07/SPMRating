"""Small desktop scorer using the public model API."""
import csv,queue,threading,tkinter as tk
from tkinter import ttk,filedialog,messagebox
from . import load_model,rate_file

def main():
    window=tk.Tk();window.title('SPM Rating v1.2.0-rc3');window.geometry('900x560')
    top=ttk.Frame(window,padding=12);top.pack(fill='x')
    ttk.Label(top,text='Model / 模型').pack(side='left');model=tk.StringVar(value='Detailed / 完整');ttk.Combobox(top,textvariable=model,values=['Detailed / 完整','Compact / 精简'],state='readonly',width=12).pack(side='left',padx=8)
    ttk.Label(top,text='Rate / 倍率').pack(side='left');rate=tk.StringVar(value='1.0');ttk.Entry(top,textvariable=rate,width=8).pack(side='left',padx=8)
    status=tk.StringVar(value='Open .osu charts, then export scores to CSV. / 选择谱面后可导出评分。');ttk.Label(window,textvariable=status,padding=12).pack(fill='x')
    table=ttk.Treeview(window,columns=('path','score','message'),show='headings')
    for key,label,width in [('path','Chart / 谱面',500),('score','Reference / 参考分数',100),('message','Status / 状态',230)]:table.heading(key,text=label);table.column(key,width=width)
    scroll=ttk.Scrollbar(window,orient='vertical',command=table.yview);table.configure(yscrollcommand=scroll.set);scroll.pack(side='right',fill='y');table.pack(fill='both',expand=True,padx=12,pady=6)
    jobs=queue.Queue();rows=[];stop=threading.Event();running=[False]
    def add():
        if running[0]:return
        paths=filedialog.askopenfilenames(filetypes=[('osu! 谱面','*.osu')])
        if not paths:return
        try:m=load_model({'Detailed / 完整':'detail','Compact / 精简':'compact'}[model.get()]);speed=float(rate.get());assert speed>0.
        except Exception as e:messagebox.showerror('Cannot start / 无法开始',str(e));return
        running[0]=True;stop.clear();status.set('Scoring; stop takes effect after the current pass. / 计算中，可停止。')
        def worker():
            try:
                for path in paths:
                    if stop.is_set():break
                    try:row=rate_file(path,m,rate=speed,scale='reference',cancel=stop.is_set)
                    except Exception as e:row=dict(path=path,error=str(e))
                    jobs.put(row)
            finally:jobs.put(None)
        threading.Thread(target=worker,daemon=True).start()
    def save():
        if not rows:return
        path=filedialog.asksaveasfilename(defaultextension='.csv',filetypes=[('CSV','*.csv')])
        if not path:return
        fields=list(dict.fromkeys(key for row in rows for key in row))
        with open(path,'w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    bottom=ttk.Frame(window,padding=12);bottom.pack(fill='x')
    ttk.Button(bottom,text='Open charts / 选择谱面',command=add).pack(side='left');ttk.Button(bottom,text='Stop / 停止',command=stop.set).pack(side='left',padx=8);ttk.Button(bottom,text='Export CSV / 导出',command=save).pack(side='left')
    def refresh():
        try:
            while True:
                row=jobs.get_nowait()
                if row is None:running[0]=False;status.set('Ready to open more charts or export. / 已结束，可继续选择谱面或导出。');continue
                rows.append(row);score='—' if 'error' in row else format(row['score'],'.3f');table.insert('',tk.END,values=(row['path'],score,row.get('error','Done / 完成')))
        except queue.Empty:pass
        window.after(100,refresh)
    refresh();window.mainloop()

if __name__=='__main__':main()
