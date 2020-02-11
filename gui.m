function varargout = gui(varargin)
% GUI MATLAB code for gui.fig
%      GUI, by itself, creates a new GUI or raises the existing
%      singleton*.
%
%      H = GUI returns the handle to a new GUI or the handle to
%      the existing singleton*.
%
%      GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUI.M with the given input arguments.
%
%      GUI('Property','Value',...) creates a new GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help gui

% Last Modified by GUIDE v2.5 01-Jan-2017 16:51:05

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @gui_OpeningFcn, ...
                   'gui_OutputFcn',  @gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before gui is made visible.
function gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to gui (see VARARGIN)

% Choose default command line output for gui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global I;                  %����һ��ȫ�ֱ���image
[filename,pathname]=uigetfile({'*.jpg';'*.png';'*.tif'},'����ͼ��');  %ѡ��ͼ��·��
str=[pathname filename];    %�ϳ�·��+�ļ���
I=imread(str);           %��ȡͼ��
axes(handles.axes1);        %ʹ�õ�һ��axes
imshow(I);               %��ʾͼ��

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global I; 
global dw; 
I1=rgb2gray(I);%����ͼת��Ϊ�Ҷ�ͼ
%figure(2),subplot(1,2,1),imshow(I1);title('�Ҷ�ͼ');
%figure(2),subplot(1,2,2),imhist(I1);title('�Ҷ�ֱ��ͼ');%���ƻҶ�ͼ��ֱ��ͼ

%ma=double(max(max(I2)));
%mi=double(min(min(I2)));
%I2=(255/(ma-mi))*I2-(255*mi)/(ma-mi);
%figure(3),imshow(I2);title('�Ҷ�����');

I2=edge(I1,'roberts',0.18,'both');%ѡ����ֵ0.18����roberts���ӽ��б�Ե���
%figure(3),imshow(I2);title('robert���ӱ�Ե���');

se=[1;1;1];
I3=imerode(I2,se);%��ͼ��ʵʩ��ʴ�����������͵ķ�����
%figure(4),imshow(I3);title('��ʴ��ͼ��');

%ƽ��ͼ��
se=strel('rectangle',[25,25]);%����ṹԪ���������ι���һ��se
I4=imclose(I3,se);% ͼ����ࡢ���ͼ��
%figure(5),imshow(I4);title('ƽ��ͼ��');	

I5=bwareaopen(I4,2500);% ȥ�����ŻҶ�ֵС��2500�Ĳ���
%figure(6),imshow(I5);title('�Ӷ������Ƴ�С����'); %��imshow������ʾ�˲���ͼ��

[y,x,z]=size(I5);%����I5��ά�ĳߴ磬�洢��x,y,z��
myI=double(I5);%��I5ת����˫����
tic      %tic��ʾ��ʱ�Ŀ�ʼ��toc��ʾ��ʱ�Ľ���
 Blue_y=zeros(y,1);%����һ��y*1������
 for i=1:y
    for j=1:x
             if(myI(i,j,1)==1) 
  %���myI(i,j,1)��myI��ͼ��������Ϊ(i,j)�ĵ�ֵΪ1�����õ�Ϊ���Ʊ�����ɫ��ɫ
  %��Blue_y(i,1)��ֵ��1
                Blue_y(i,1)= Blue_y(i,1)+1;%��ɫ���ص�ͳ�� 
            end  
     end 
 end
 [temp MaxY]=max(Blue_y);%Y����������ȷ��
  %tempΪ����Blue_y��Ԫ���е����ֵ��MaxYΪ��ֵ������
 Y1=MaxY;
 while ((Blue_y(Y1,1)>=12)&&(Y1>1))
        Y1=Y1-1;
 end
 Y2=MaxY;
 while ((Blue_y(Y2,1)>=12)&&(Y2<y))
        Y2=Y2+1;
 end
 IY=I(Y1:Y2,:,:);
%IYΪԭʼͼ��I�н�ȡ����������Y1��Y2֮��Ĳ���
%�з���������ȷ��
%��һ��ȷ��x����ĳ�������
 Blue_x=zeros(1,x);
 for j=1:x
     for i=Y1:Y2
            if(myI(i,j,1)==1)
                Blue_x(1,j)= Blue_x(1,j)+1;               
            end  
     end       
 end
 
 X1=1;
 while ((Blue_x(1,X1)<3)&&(X1<x))
       X1=X1+1;
 end
 X2=x;
 while ((Blue_x(1,X2)<3)&&(X2>X1))
        X2=X2-1;
 end
 X1=X1-1;%�Գ��������У��
 X2=X2+1;
  dw=I(Y1:Y2,X1:X2,:);
  %dw=I(Y1:Y2,:,:);
 t=toc; 
%figure(7),subplot(1,2,1),imshow(IY),title('�з����������');%�з���������ȷ��
%figure(7),subplot(1,2,2),imshow(dw),title('��λ���к�Ĳ�ɫ����ͼ��');%��λ��ĳ�������������ʾ��
axes(handles.axes2);        %ʹ�õڶ���axes
imshow(dw);               %��ʾͼ��

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global dw;
b=rgb2gray(dw);%������ͼ��ת��Ϊ�Ҷ�ͼ
%imwrite(b,'���ƻҶ�ͼ��.jpg');%���Ҷ�ͼ��д���ļ���
%figure(8);subplot(3,2,1),imshow(b),title('���ƻҶ�ͼ��')
g_max=double(max(max(b)));
g_min=double(min(min(b)));
T=round(g_max-(g_max-g_min)/3); % TΪ��ֵ������ֵ
d=(double(b)>=T);  % d:��ֵͼ��
%imwrite(d,'���ƶ�ֵͼ��.jpg');
%subplot(3,2,2),imshow(d),title('��ֵ�˲�ǰ')%��ֵ�˲�ǰ

%{
% ��ת  
rotate=0;  
bw=edge(d);  
[m,n]=size(d);  
theta=1:179;  
% bw ��ʾ��Ҫ�任��ͼ��theta ��ʾ�任�ĽǶ�  
% ����ֵ r ��ʾ�����а����˶�Ӧ�� theta��ÿһ���Ƕȵ� Radon �任���  
% ���� xp ������Ӧ���� x�������  
[r,xp]=radon(bw,theta);  
i=find(r>0);  
[foo,ind]=sort(-r(i));  
k=i(ind(1:size(i)));  
[y,x]=ind2sub(size(r),k);  
[mm,nn]=size(x);  
if mm~=0 && nn~=0  
    j=1;  
    while mm~=1 && j<180 && nn~=0  
        i=find(r>j);  
        [foo,ind]=sort(-r(i));  
        k=i(ind(1:size(i)));  
        [y,x]=ind2sub(size(r),k);  
        [mm,nn]=size(x);  
        j=j+1;  
    end  
    if nn~=0  
        if x   % Enpty matrix: 0-by-1 when x is an enpty array.  
            x=x;  
        else  % ���� x Ϊ��ֵ  
            x=90; % ��ʵ���ǲ���ת  
        end  
        d=imrotate(d,abs(90-x)); % ��תͼ��  
        rotate=1;  
    end  
end  
imwrite(d,'4.Radon �任��ת��Ķ�ֵͼ��.jpg');  
figure(8),subplot(3,2,2),imshow(d),title('4.Radon �任��ת��Ķ�ֵͼ��');  
%}

%�˲�
h=fspecial('average',3);
%����Ԥ������˲����ӣ�averageΪ��ֵ�˲���ģ��ĳߴ�Ϊ3*3
d=im2bw(round(filter2(h,d)));%ʹ��ָ�����˲���h��h����d����ֵ�˲�
%imwrite(d,'��ֵ�˲���.jpg');
%subplot(3,2,3),imshow(d),title('��ֵ�˲���')
% ĳЩͼ����в���
% ���ͻ�ʴ
% se=strel('square',3);  % ʹ��һ��3X3�������ν��Ԫ�ض���Դ�����ͼ���������
% 'line'/'diamond'/'ball'...
se=eye(2); % eye(n) returns the n-by-n identity matrix ��λ����
[m,n]=size(d);%���ؾ���b�ĳߴ���Ϣ�� ���洢��m,n��
if bwarea(d)/m/n>=0.365 %�����ֵͼ���ж�������������������ı��Ƿ����0.365
    d=imerode(d,se);%�������0.365��ͼ����и�ʴ
elseif bwarea(d)/m/n<=0.235 %�����ֵͼ���ж�������������������ı��Ƿ�С��0.235
    d=imdilate(d,se);%���С����ʵ�����Ͳ���
end
%imwrite(d,'���ͻ�ʴ�����.jpg');
%subplot(3,2,4),imshow(d),title('���ͻ�ʴ�����');

%Ѱ�����������ֵĿ飬�����ȴ���ĳ��ֵ������Ϊ�ÿ��������ַ���ɣ���Ҫ�ָ�
 d=qiege(d);
[m,n]=size(d);
%subplot(3,2,5),imshow(d),title(n)
k1=1;k2=1;s=sum(d);j=1;
while j~=n
    while s(j)==0
        j=j+1;
    end
    k1=j;
    while s(j)~=0 && j<=n-1
        j=j+1;
    end
    k2=j-1;
    if k2-k1>=round(n/6.5)
        [val,num]=min(sum(d(:,[k1+5:k2-5])));
        d(:,k1+num+5)=0;  % �ָ�
    end
end
% ���и�
d=qiege(d);
% �и�� 7 ���ַ�
y1=10;y2=0.25;flag=0;word1=[];
while flag==0
    [m,n]=size(d);
    left=1;wide=0;
    while sum(d(:,wide+1))~=0
        wide=wide+1;
    end
    if wide<y1   % ��Ϊ��������
        d(:,[1:wide])=0;
        d=qiege(d);
    else
        temp=qiege(imcrop(d,[1 1 wide m]));
        [m,n]=size(temp);
        all=sum(sum(temp));
        two_thirds=sum(sum(temp([round(m/3):2*round(m/3)],:)));
        if two_thirds/all>y2
            flag=1;word1=temp;   % WORD 1
        end
        d(:,[1:wide])=0;d=qiege(d);
    end
end
% �ָ���ڶ������߸��ַ�
[word2,d]=getword(d);
[word3,d]=getword(d);
[word4,d]=getword(d);
[word5,d]=getword(d);
[word6,d]=getword(d);
[word7,d]=getword(d);

[m,n]=size(word1);
% ����ϵͳ�����й�һ����СΪ 40*20,�˴���ʾ
word1=imresize(word1,[40 20]);
word2=imresize(word2,[40 20]);
word3=imresize(word3,[40 20]);
word4=imresize(word4,[40 20]);
word5=imresize(word5,[40 20]);
word6=imresize(word6,[40 20]);
word7=imresize(word7,[40 20]);

%subplot(2,7,8),imshow(word1),title('1');
%subplot(2,7,9),imshow(word2),title('2');
%subplot(2,7,10),imshow(word3),title('3');
%subplot(2,7,11),imshow(word4),title('4');
%subplot(2,7,12),imshow(word5),title('5');
%subplot(2,7,13),imshow(word6),title('6');
%subplot(2,7,14),imshow(word7),title('7');
imwrite(word1,'1.jpg');
imwrite(word2,'2.jpg');
imwrite(word3,'3.jpg');
imwrite(word4,'4.jpg');
imwrite(word5,'5.jpg');
imwrite(word6,'6.jpg');
imwrite(word7,'7.jpg');

liccode=char(['0':'9' 'A':'Z' '����³����ԥ��']);  %�����Զ�ʶ���ַ������  
SubBw2=zeros(40,20);  %����40*20��ȫ0����
l=1;
for I=1:7
      ii=int2str(I);  %תΪ��
       t=imread([ii,'.jpg']); %��ȡͼƬ�ļ��е�����
      SegBw2=imresize(t,[32 16],'nearest'); %��ͼ�������Ŵ���
       SegBw2=double(SegBw2)>16;
        if l==1                 %��һλ����ʶ��
            kmin=37;
            kmax=43;
        elseif l==2             %�ڶ�λ A~Z ��ĸʶ��
            kmin=11;
            kmax=36;
        else l>=3               %����λ�Ժ�����ĸ������ʶ��
            kmin=1;
            kmax=36;
        end
        
        for k2=kmin:kmax
            fname=strcat('sample/',liccode(k2),'.bmp'); %��������ת�����ַ���
            SamBw2 = imread(fname);
           SamBw2=double(SamBw2)>1;
           %size(SamBw2)
            for  i=1:32
                for j=1:16
                    SubBw2(i,j)=SegBw2(i,j)-SamBw2(i,j);
                end
            end
           % �����൱������ͼ����õ�������ͼ
            Dmax=0;
            for k1=1:32
                for l1=1:16
                    if  ( SubBw2(k1,l1) > 0 || SubBw2(k1,l1) <0 )
                        Dmax=Dmax+1;
                    end
                end
            end
            Error(k2)=Dmax;
        end
        Error1=Error(kmin:kmax);
        MinError=min(Error1);
        findc=find(Error1==MinError);
        Code(l*2-1)=liccode(findc(1)+kmin-1);
        Code(l*2)=' '; %���������ͼ��
        l=l+1;
end
%figure(10),imshow(dw),title (['���ƺ���:', Code],'Color','r');
set(handles.text1,'string',num2str(Code));
