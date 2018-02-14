function [learnData]=extractLearningData()
%%This function extracts all the learning data and downsamples the images
%%as well as greyscales them

classes = {'Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes', ...
    'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles', ...
    'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles', ...
    'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency', ...
    'Military', 'Commercial', 'Trains'};

files = dir('C:\Users\kmh43\599 perception project\deploy\trainval\*\*_image.jpg');
learningBoxes=cell(round(numel(files)/3),2);
imgIdx=1;
for countFile=1:3:numel(files)
    
    snapshot = [files(countFile).folder, '/', files(countFile).name];
    disp(snapshot)
    
    img = rgb2gray(imread(snapshot));
    img=imresize(img, 0.5); %resize img
    fileName=[files(countFile).folder(find(files(1).folder=='\',1,'last')+1:end) '_' files(countFile).name];
    rsPath='C:\Users\kmh43\Documents\Fall 2017\EECS 498\training_data\trainval_resize';
    learningBoxes{imgIdx,1}=[rsPath  '\' fileName]; %save file name
     imwrite(img,[rsPath  '\' fileName])   %save resized img

    
    xyz = memmapfile(strrep(snapshot, '_image.jpg', '_cloud.bin'), ...
        'format', 'single').Data;
    xyz = reshape(xyz, [numel(xyz) / 3, 3])';
    
    proj = memmapfile(strrep(snapshot, '_image.jpg', '_proj.bin'), ...
        'format', 'single').Data;
    proj = reshape(proj, [4, 3])';
    
    try
        bbox = memmapfile(strrep(snapshot, '_image.jpg', '_bbox.bin'), ...
            'format', 'single').Data;
    catch
        disp('[*] no bbox found.')
        bbox = single([]);
    end
    bbox = reshape(bbox, [11, numel(bbox) / 11])';
    
    uv = proj * [xyz; ones(1, size(xyz, 2))];
    uv = uv ./ uv(3, :);
    
        clr = sqrt(sum(xyz.^2, 1));
       
    
    colors =[0, 0.4470, 0.7410
        0.8500, 0.3250, 0.0980
        0.9290, 0.6940, 0.1250
        0.4940, 0.1840, 0.5560
        0.4660, 0.6740, 0.1880
        0.3010, 0.7450, 0.9330
        0.6350, 0.0780, 0.1840];
    
    carNum=0;
    
    
    for k = 1:size(bbox, 1)
        b = bbox(k, :);
        c = classes{int64(b(10)) + 1};
        
        if b(11)==0
            
            carNum=carNum+1;
            
            n = b(1:3);
            theta = norm(n, 2);
            n = n / theta;
            R = rot(n, theta);
            t = reshape(b(4:6), [3, 1]);
            
            sz = b(7:9);
            [vert_3D, edges] = get_bbox(-sz / 2, sz / 2);
            vert_3D = R * vert_3D + t;
            
            vert_2D = proj * [vert_3D; ones(1, 8)];
            vert_2D = vert_2D ./ vert_2D(3, :);
            
            clr = colors(mod(k - 1, size(colors, 1)) + 1, :);
                        
            zDir=vert_3D(3,3)-vert_3D(3,1);
            xDir=vert_3D(1,7)-vert_3D(1,3);
            
            coln=2;
            
            % Get 2D bbox coordinates and w,h
            boxXY=round(min(vert_2D,[],2)/2);
            boxXY(boxXY<0)=1; %prevent from getting negative pixel locations
            boxWH=round(max(vert_2D,[],2)/2)-boxXY;
            
            if boxXY(1)+boxWH(1)>size(img,2)
                boxWH(1)=size(img,2)-boxXY(1);
            end
            
            
            learningBoxes{imgIdx,coln}=[learningBoxes{imgIdx,coln};boxXY(1),boxXY(2),boxWH(1),boxWH(2)];
        end
    end
    imgIdx=imgIdx+1;
end
learnData=table(learningBoxes(:,1),learningBoxes(:,2),'VariableNames',{'File','car'});
save('learnData_1_3.mat','learnData')
end


function [v, e] = get_bbox(p1, p2)
v = [p1(1), p1(1), p1(1), p1(1), p2(1), p2(1), p2(1), p2(1)
    p1(2), p1(2), p2(2), p2(2), p1(2), p1(2), p2(2), p2(2)
    p1(3), p2(3), p1(3), p2(3), p1(3), p2(3), p1(3), p2(3)];
e = [3, 4, 1, 1, 4, 4, 1, 2, 3, 4, 5, 5, 8, 8
    8, 7, 2, 3, 2, 3, 5, 6, 7, 8, 6, 7, 6, 7];
end


function R = rot(n, theta)
n = n / norm(n, 2);
K = [0, -n(3), n(2); n(3), 0, -n(1); -n(2), n(1), 0];
R = eye(3) + sin(theta) * K + (1 - cos(theta)) * K^2;
end
