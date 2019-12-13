from Data import Data
import pandas as pd


def teacher_fx1(result,labelCount):
    #统计结果中每个老师的学生类别分布
    #result.SelectTopN(10)
    teacher={'1':[0]*labelCount,'2':[0]*labelCount,'3':[0]*labelCount}
    for i in range(len(result.data)):
        #print(result.data[i])
        teacherid=str(result.data[i][0])
        #该评价所属类别是result.predict[i]，该teacher的对应的评价数量加1
        #print('result.predict',result.predict[i])
        teacher[teacherid][result.predict[i]]+=1
    print("teacher:")
    for i in ['1','2','3']:
        print('teacher_id:',i,'comments_distri:',teacher[i])
            
def teacher_fx2(data,teacher_id):
    #统计每个问题的答案分布情况
    list1=['instr','class',	'nb.repeat','attendance','difficulty','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20','Q21','Q22','Q23','Q24	','Q25','Q26','Q27','Q28']
    df=pd.DataFrame(data,columns=list1)
    df=df.loc[df['instr']== int(teacher_id)]
    print("该教师答案得分小于3的问题如下:") 
    for col in df.columns:
        if df[col].mean() <3:
            print('col',col,"该问题答案的均值为%.2f"  %df[col].mean())    #计算每列均值

def class_fx1(result,labelCount):
    class1={'1':[0]*labelCount,'2':[0]*labelCount,'3':[0]*labelCount,'4':[0]*labelCount,'5':[0]*labelCount,'6':[0]*labelCount,'7':[0]*labelCount,'8':[0]*labelCount,'9':[0]*labelCount,'10':[0]*labelCount,'11':[0]*labelCount,'12':[0]*labelCount,'13':[0]*labelCount}
    for i in range(len(result.data)):
        #print(result.data[i])
        class_id=str(result.data[i][1])
        class1[class_id][result.predict[i]]+=1
    print("class:")
    for i in ['1','2','3','4','5','6','7','8','9','10','11','12','13']:
        print('class_id:',i,'comments_distri:',class1[i])

def class_fx2(data,class_id):
    list1=['instr','class',	'nb.repeat','attendance','difficulty','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20','Q21','Q22','Q23','Q24	','Q25','Q26','Q27','Q28']
    df=pd.DataFrame(data,columns=list1)
    df=df.loc[df['class'] == int(class_id)]
    print("该课程答案得分小于3的问题如下:") 
    for col in df.columns:
        if df[col].mean() <3:
            print('col',col,"答案均值为%.2f"  %df[col].mean())    #计算每列均值
    

test = Data()
test.ReadData('data.csv')
result = test.KMeans(2)
labelCount=result.ShowLabelInfo()
teacher_fx1(result,labelCount)
class_fx1(result,labelCount)
teacher_fx2(result.data,'3')
#class_fx2(result.data,'13')
