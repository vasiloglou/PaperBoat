function tofl(filename, x, labels, attributes, metadata)
% Utility to save data in the Analytics 1305 fl-lite format
% This software is provided by the Analytics 1305 "as is" 
% without any warranty.
%
% tofl(filename, x, labels, attribute_precision, metadata)
%
% filename: name of the file you want save the data
% x: a matrix that  contains the data. Each COLUMN is a point
%    and every row is a feature. We prefered this representation
%    since Matlab operations (specially sparse) are much faster
% labels: is optional, you can provide labels for the columns if you want
%         in a cell array, i.e: 
%
%                   labels={'height' 'weight'} 
% attributes: fl-lite has the capability to store variables in
%             different precisions. So attribute_precision is 
%             a struct array. Each element has a property called
%             precision and a property called features, indicating
%             which features have that precision and a property
%             called storage that can be either 'dense' or
%             'sparse'. The following precisions are supported
%             'double, float, int, bool'. Here is an example of attributes
%              attributes(1).precision='double';
%            
%                 attributes(1).storage='dense';
%                 attributes(1).features=1:5;
%                 attributes(2).precision='double';
%                 attributes(2).storage='sparse';
%                 attributes(2).features=6:10;
%
% metadata: It is possible to store some extra information with every point
%           that doesn't participate in the numerical computation between 
%           points. It is treated separately. Metadata can be class 
%           information for classification problems or target 
%           values for regression, or just a point id. We currently support
%           3 precisions: signed char, double, int. You can only store 3
%           3 numbers as metadata (one for int, one for signed char and one
%           for double). Metadata is a vector of length 3. Each element
%           refers to the column that will be used from x matrix as
%           metadata. The first one is signed char, the second one is
%           double and the third is int. If you don't want to use any
%           column from x set the appropriate element of metadata to 0
%           For example metadata=[3 0 0] will pick the 3 colums of x and
%           it will store it as a signed char.
%
%           Here is a complete example for storing a matrix in dense double
%           and sparse double format
%
%           x=[rand(10,100) ; sprand(1000, 100, 0.1)];
%           labels={};
%           attributes(1).precision='double';
%           attributes(1).storage='dense';
%           attributes(1).features=2:10;
%           attributes(2).precision='double';
%           attributes(2).storage='sparse';
%           attributes(2).features=11:1010;
%           metadata=[1 0 0];
%           tofl('temp.fl', x, labels, attributes, metadata);



% open the file
fid=fopen(filename, 'w');
% write the header
fprintf(fid, 'header ');
% for this specific file format we have only 3 metadata
fprintf(fid, 'meta:3 ');
% now iterate over the attributes to find the precisions
% also check if the input is right

dense_precisions=[];
sparse_precisions=[];
for i=1:length(attributes)
  if ~(strcmp(attributes(i).storage, 'dense') ... 
       | strcmp(attributes(i).storage, 'sparse'))
    error(['You can only store data in sparse or dense format '...
           'you attempted to store data in ' attributes(i).storage ...
           'format.']);
  else
    if ~(strcmp(attributes(i).precision,'double') ... 
       | strcmp(attributes(i).precision,'float') ...
       | strcmp(attributes(i).precision,'int') ...
       | strcmp(attributes(i).precision,'bool'))
      error(['You can only save data in double, float, int or bool format '...
             'This precision ' attributes(i).precision ' you entered is '
             'not valid.']);
    else 
      if strcmp(attributes(i).storage,'dense')
        dense_precisions(end+1).precision=attributes(i).precision;
        dense_precisions(end).features=attributes(i).features;
      else
        sparse_precisions(end+1).precision=attributes(i).precision;
        sparse_precisions(end).features=attributes(i).features;
      end
    end
  end
end

% check that metadata is correct
if (length(metadata)~=3)
  error('the length of metadata must be 3');
end
for i=1:3
  if (metadata(i)<0 | metadata(i)>size(x,1))
      error(['metadata elements must be between 0 and size(x, 2).'...
          'The ' num2str(i) 'th element of metadata is ' ...
          num2str(metadata(i)) ' which is not in [0,' num2str(size(x,1))]);
  end
end

if (length(dense_precisions) > 1) ...
    | (length(sparse_precisions) > 1) 
  error('This version supports only one dense and one sparse precision');
end  

for i=1:length(dense_precisions)
  fprintf(fid, [dense_precisions(i).precision ':'...
      num2str(length(dense_precisions(i).features)) ' ']);
end

for i=1:length(sparse_precisions)
  fprintf(fid, ['sparse:' sparse_precisions(i).precision ':'...
      num2str(length(sparse_precisions(i).features)) ' ']);
end

%change line
fprintf(fid, '\n');
%if there are labels we have to print them
if (length(labels)~=0)
  fprintf(fid, 'labels ');
  for i=1:length(labels)
    fprintf(fid, [labels{i} ' ']);
  end
  fprintf(fid, '\n');
end

%Now it is time to store the data
for i=1:size(x,2)
  %store the metadata first
  for j=1:length(metadata)
    if (metadata(j)==0)
      fprintf(fid, '0 ');
    else
      if (j==1)
        fprintf(fid, '%i ', int8(x(metadata(j), i)));
      else 
        if (j==2) 
          fprintf(fid, '%g ', x(metadata(j), i));  
        else
          if (j==3)
            fprintf(fid, '%i ', int64(x(metadata(j), i)));
          end
        end
      end
    end
  end
  offset=0;
  %write the dense first
  for k=1:length(dense_precisions)
    if (strcmp(dense_precisions(k).precision,'double') ... 
        | strcmp(dense_precisions(k).precision,'float'))
      for j=1:length(dense_precisions(k).features)
        ind=dense_precisions(k).features(j);
        fprintf(fid, '%g ', x(ind, i)); 
      end
    elseif (strcmp(dense_precisions(k).precision,'int') ...
        | strcmp(dense_precisions(k).precision,'bool'))
      for j=1:length(dense_precisions(k).features)
        ind=dense_precisions(k).features(j);
        fprintf(fid, '%i ', x(ind, i)); 
      end
    end
        
    offset=offset+length(dense_precisions(k).features);
  end
  %then write the sparse part
  for k=1:length(sparse_precisions)
    I=find(x(sparse_precisions(k).features, i));
    if (strcmp(sparse_precisions(k).precision,'double') ...
        |strcmp(sparse_precisions(k).precision,'float' ))  
      if length(sparse_precisions==1)
        for j=1:length(I)
          ind=I(j)+offset;
          fprintf(fid, '%i:%g ',ind, x(sparse_precisions(k).features(I(j)), i)); 
        end
      else
        for j=1:length(I)
          ind=I(j)+offset;
          fprintf(fid, '%s:%i:%g ', sparse_precisions(k).precision, ind, x(sparse_precisions(k).features(I(j)), i)); 
        end
      end
      offset=offset+length(sparse_precisions(k).features);
    else    
      if (strcmp(sparse_precisions(k).precision,'int') ...
        | strcmp(sparse_precisions(k).precision,'bool'))
        if length(sparse_precisions==1)
          for j=1:length(I)
            ind=I(j)+offset;
            fprintf(fid, '%i:%i ',ind, x(sparse_precisions(k).features(I(j)), i)); 
          end
        else
          for j=1:length(I)
            ind=I(j)+offset;
            fprintf(fid, '%s:%i:%i ',sparse_precisions(k).precision, ind, x(sparse_precisions(k).features(I(j)), i)); 
          end
        end
       offset=offset+length(sparse_precisions(k).features);
      end
    end
    fprintf(fid,'\n');
  end
end
fclose(fid);
