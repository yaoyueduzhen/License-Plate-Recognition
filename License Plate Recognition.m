I=imread('picture/car3.jpg');
figure(1),imshow(I);title('原图');%将车牌的原图显示出来

I1=rgb2gray(I);%将彩图转换为灰度图
figure(2),subplot(1,2,1),imshow(I1);title('灰度图');
figure(2),subplot(1,2,2),imhist(I1);title('灰度直方图');%绘制灰度图的直方图

%ma=double(max(max(I2)));
%mi=double(min(min(I2)));
%I2=(255/(ma-mi))*I2-(255*mi)/(ma-mi);
%figure(3),imshow(I2);title('灰度拉伸');

I2=edge(I1,'roberts',0.18,'both');%选择阈值0.18，用roberts算子进行边缘检测
figure(3),imshow(I2);title('robert算子边缘检测');

se=[1;1;1];
I3=imerode(I2,se);%对图像实施腐蚀操作，即膨胀的反操作
figure(4),imshow(I3);title('腐蚀后图像');

%平滑图像
se=strel('rectangle',[25,25]);%构造结构元素以正方形构造一个se
I4=imclose(I3,se);% 图像聚类、填充图像
figure(5),imshow(I4);title('平滑图像');	

 I5=bwareaopen(I4,2500);% 去除聚团灰度值小于2000的部分
figure(6),imshow(I5);title('从对象中移除小对象'); %用imshow函数显示滤波后图像

[y,x,z]=size(I5);%返回I5各维的尺寸，存储在x,y,z中
myI=double(I5);%将I5转换成双精度
tic      %tic表示计时的开始，toc表示计时的结束
 Blue_y=zeros(y,1);%产生一个y*1的零阵
 for i=1:y
    for j=1:x
             if(myI(i,j,1)==1) 
  %如果myI(i,j,1)即myI的图像中坐标为(i,j)的点值为1，即该点为车牌背景颜色蓝色
  %则Blue_y(i,1)的值加1
                Blue_y(i,1)= Blue_y(i,1)+1;%蓝色像素点统计 
            end  
     end 
 end
 [temp MaxY]=max(Blue_y);%Y方向车牌区域确定
  %temp为向量Blue_y的元素中的最大值，MaxY为该值的索引
 PY1=MaxY;
 while ((Blue_y(PY1,1)>=12)&&(PY1>1))
        PY1=PY1-1;
 end    
 PY2=MaxY;
 while ((Blue_y(PY2,1)>=12)&&(PY2<y))
        PY2=PY2+1;
 end
 IY=I(PY1:PY2,:,:);
%IY为原始图像I中截取的纵坐标在PY1：PY2之间的部分
%行方向车牌区域确定
%进一步确定x方向的车牌区域
 Blue_x=zeros(1,x);
 for j=1:x
     for i=PY1:PY2
            if(myI(i,j,1)==1)
                Blue_x(1,j)= Blue_x(1,j)+1;               
            end  
     end       
 end
  
 PX1=1;
 while ((Blue_x(1,PX1)<3)&&(PX1<x))
       PX1=PX1+1;
 end    
 PX2=x;
 while ((Blue_x(1,PX2)<3)&&(PX2>PX1))
        PX2=PX2-1;
 end
 PX1=PX1-1;%对车牌区域的校正
 PX2=PX2+1;
  dw=I(PY1:PY2,PX1:PX2,:);
  %dw=I(PY1:PY2,:,:);
 t=toc; 
figure(7),subplot(1,2,1),imshow(IY),title('行方向合理区域');%行方向车牌区域确定
figure(7),subplot(1,2,2),imshow(dw),title('定位剪切后的彩色车牌图像');%定位后的车牌区域如下所示：



imwrite(dw,'dw.jpg');%将彩色车牌写入dw文件中
a=imread('dw.jpg');%读取车牌文件中的数据
b=rgb2gray(a);%将车牌图像转换为灰度图
imwrite(b,'车牌灰度图像.jpg');%将灰度图像写入文件中
figure(8);subplot(3,2,1),imshow(b),title('车牌灰度图像')
g_max=double(max(max(b)));
g_min=double(min(min(b)));
T=round(g_max-(g_max-g_min)/3); % T为二值化的阈值
d=(double(b)>=T);  % d:二值图像
imwrite(d,'车牌二值图像.jpg');
subplot(3,2,2),imshow(d),title('均值滤波前')%均值滤波前

%{
% 旋转  
rotate=0;  
bw=edge(d);  
[m,n]=size(d);  
theta=1:179;  
% bw 表示需要变换的图像，theta 表示变换的角度  
% 返回值 r 表示的列中包含了对应于 theta中每一个角度的 Radon 变换结果  
% 向量 xp 包含相应的沿 x轴的坐标  
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
        else  % 可能 x 为空值  
            x=90; % 其实就是不旋转  
        end  
        d=imrotate(d,abs(90-x)); % 旋转图像  
        rotate=1;  
    end  
end  
imwrite(d,'4.Radon 变换旋转后的二值图像.jpg');  
figure(8),subplot(3,2,2),imshow(d),title('4.Radon 变换旋转后的二值图像');  
%}

%滤波
h=fspecial('average',3);
%建立预定义的滤波算子，average为均值滤波，模板的尺寸为3*3
d=im2bw(round(filter2(h,d)));%使用指定的滤波器h对h进行d即均值滤波
imwrite(d,'均值滤波后.jpg');
subplot(3,2,3),imshow(d),title('均值滤波后')
% 某些图像进行操作
% 膨胀或腐蚀
% se=strel('square',3);  % 使用一个3X3的正方形结果元素对象对创建的图像进行膨胀
% 'line'/'diamond'/'ball'...
se=eye(2); % eye(n) returns the n-by-n identity matrix 单位矩阵
[m,n]=size(d);%返回矩阵b的尺寸信息， 并存储在m,n中
if bwarea(d)/m/n>=0.365 %计算二值图像中对象的总面积与整个面积的比是否大于0.365
    d=imerode(d,se);%如果大于0.365则图像进行腐蚀
elseif bwarea(d)/m/n<=0.235 %计算二值图像中对象的总面积与整个面积的比是否小于0.235
    d=imdilate(d,se);%如果小于则实现膨胀操作
end
imwrite(d,'膨胀或腐蚀处理后.jpg');
subplot(3,2,4),imshow(d),title('膨胀或腐蚀处理后');

%寻找连续有文字的块，若长度大于某阈值，则认为该块有两个字符组成，需要分割
 d=qiege(d);
[m,n]=size(d);
subplot(3,2,5),imshow(d),title(n)
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
        d(:,k1+num+5)=0;  % 分割
    end
end
% 再切割
d=qiege(d);
% 切割出 7 个字符
y1=10;y2=0.25;flag=0;word1=[];
while flag==0
    [m,n]=size(d);
    left=1;wide=0;
    while sum(d(:,wide+1))~=0
        wide=wide+1;
    end
    if wide<y1   % 认为是左侧干扰
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
% 分割出第二个字符
[word2,d]=getword(d);
% 分割出第三个字符
[word3,d]=getword(d);
% 分割出第四个字符
[word4,d]=getword(d);
% 分割出第五个字符
[word5,d]=getword(d);
% 分割出第六个字符
[word6,d]=getword(d);
% 分割出第七个字符
[word7,d]=getword(d);
figure(9);
subplot(2,7,1),imshow(word1),title('1');
subplot(2,7,2),imshow(word2),title('2');
subplot(2,7,3),imshow(word3),title('3');
subplot(2,7,4),imshow(word4),title('4');
subplot(2,7,5),imshow(word5),title('5');
subplot(2,7,6),imshow(word6),title('6');
subplot(2,7,7),imshow(word7),title('7');
[m,n]=size(word1);
% 商用系统程序中归一化大小为 40*20,此处演示
word1=imresize(word1,[40 20]);
word2=imresize(word2,[40 20]);
word3=imresize(word3,[40 20]);
word4=imresize(word4,[40 20]);
word5=imresize(word5,[40 20]);
word6=imresize(word6,[40 20]);
word7=imresize(word7,[40 20]);

subplot(2,7,8),imshow(word1),title('1');
subplot(2,7,9),imshow(word2),title('2');
subplot(2,7,10),imshow(word3),title('3');
subplot(2,7,11),imshow(word4),title('4');
subplot(2,7,12),imshow(word5),title('5');
subplot(2,7,13),imshow(word6),title('6');
subplot(2,7,14),imshow(word7),title('7');
imwrite(word1,'1.jpg');
imwrite(word2,'2.jpg');
imwrite(word3,'3.jpg');
imwrite(word4,'4.jpg');
imwrite(word5,'5.jpg');
imwrite(word6,'6.jpg');
imwrite(word7,'7.jpg');

liccode=char(['0':'9' 'A':'Z' '京辽鲁陕苏豫浙']);  %建立自动识别字符代码表  
SubBw2=zeros(40,20);  %产生40*20的全0矩阵
l=1;
for I=1:7
      ii=int2str(I);  %转为串
     t=imread([ii,'.jpg']); %读取图片文件中的数据
      SegBw2=imresize(t,[32 16],'nearest'); %对图像做缩放处理
       SegBw2=double(SegBw2)>16;
        if l==1                 %第一位汉字识别
            kmin=37;
            kmax=43;
        elseif l==2             %第二位 A~Z 字母识别
            kmin=11;
            kmax=36;
        else l>=3               %第三位以后是字母或数字识别
            kmin=1;
            kmax=36;
        end
        
        for k2=kmin:kmax
            fname=strcat('sample/',liccode(k2),'.bmp'); %把行向量转化成字符串
            SamBw2 = imread(fname);
           SamBw2=double(SamBw2)>1;
           %size(SamBw2)
            for  i=1:32
                for j=1:16
                    SubBw2(i,j)=SegBw2(i,j)-SamBw2(i,j);
                end
            end
           % 以上相当于两幅图相减得到第三幅图
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
        Code(l*2)=' '; %输出最大相关图像
        l=l+1;
end
figure(10),imshow(dw),title (['车牌号码:', Code],'Color','r');

