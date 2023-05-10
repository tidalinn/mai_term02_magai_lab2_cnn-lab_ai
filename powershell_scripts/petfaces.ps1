$source = 'http://www.soshnikov.com/permanent/data/petfaces.tar.gz'
$folderPath = 'D:\EDUCATION\01_mai\magai_lab2_cnn-lab_ai'
$destination = $folderPath + '\data'
$file = 'petfaces.tar.gz'

cd $destination
Invoke-WebRequest -Uri $source -OutFile $file
tar xfz $file
rm $file