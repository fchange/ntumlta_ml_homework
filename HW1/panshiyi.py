# ��ʯ�ٵĴ��룬2020��1��16�ա�
# ������ǰ�²�һ�����н���ɡ�
import turtle

p = turtle.Pen()
p.speed(0)
p.pencolor("red")
p.fillcolor("yellow")
p.begin_fill()
for i in range(36):
    for j in range(4):
        p.forward(100)
        p.left(90)
    p.right(10)
p.end_fill()