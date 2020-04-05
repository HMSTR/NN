uses graphABC,utils;
const FRAMES_MAX = 164;
const FRAMES_MIN = 0;
const dir = 'data\10';
var path:string = 'F:\AI\MRI\RAW\';
b:array[FRAMES_MIN..FRAMES_MAX] of Picture;
frameColor:color;
x:integer;
currentFrame:integer;
var w,h:integer;
var box: record
  x1,x2,y1,y2:integer;
end;

procedure updateAll(tu:integer:=0);
begin
b[currentFrame].Draw(0,0);
SetPenWidth(1);
setpencolor(frameColor);
DrawRectangle(box.x1,box.y1,box.x2,box.y2);
if(tu=1) then
  begin
    textout(w-200,60,'Selected box coords:'+box.x1.ToString()+'x'+box.y1.ToString());
    textout(w-200,75,'size:'+(box.x2-box.x1).ToString()+'x'+(box.y2-box.y1).ToString())
  end;
end;

procedure rec();
var tmp:Picture;
var frame,w,h,i,j:integer;
var pix:color;
begin
  CreateDir(dir);
  for frame:=FRAMES_MIN to FRAMES_MAX do
    Begin
      w:=box.x2-box.x1;
      h:=box.y2-box.y1;
      tmp := Picture.Create(w+1,h+1);
      frameColor:=clRed;
      for i:=0 to w Do
        for j:=0 to h Do
          Begin
            pix := b[frame].GetPixel(box.x1+i,box.y1+j);
            tmp.SetPixel(i,j,pix);
          End;
        currentFrame:=frame;
        updateAll();
      tmp.Save(dir+'\'+frame.ToString()+'.jpg');
    End;
    frameColor:=clGreen;
end;

procedure keyboard(key: integer);
begin
case key of
  VK_UP:
    begin
      if(currentFrame<FRAMES_MAX) then
        currentFrame+=1;
    end;
  VK_DOWN:
    begin
      if(currentFrame>FRAMES_MIN) then
        currentFrame-=1;
    end;
  VK_RIGHT:
    rec();
end;
updateAll();
end;


procedure mouse(x,y,m: integer);
begin
  case m of
    1:
      begin
      // left
      if(x>16)and(x<b[currentFrame].Width-16)and(y>16)and(y<b[currentFrame].height-16) then
      Begin
        box.x1 := x-16;
        box.y1 := y-16;
        box.x2 := x+16;
        box.y2 := y+16;
        End;
      end;
  end;
  if(box.x1*box.x2*box.y1*box.y2<>0) then
  begin
    updateAll(1);
  end;
end;



Begin
frameColor:=clGreen;
w:=windowwidth();
h:=windowheight();
currentFrame:=1;
for x:= FRAMES_MIN to FRAMES_MAX do
  begin
    b[x] := Picture.Create(path+x.ToString()+'.jpg');
    b[x].Load(path+x.ToString()+'.jpg');
  end;
updateAll();
textout(w-200,0,'set the rectangle to save right.');
textout(w-200,40,'When ready, press RIGHT ARROW');
OnKeyDown:=keyboard;
OnMouseDown:=mouse;
End.

