var fd;
var xhr;

$(document).ready(function(){
    $('#driver-uploader-path').change(function(){
        
        //alert("Len", this.files.length)
        /*
        var len = this.files.length
        for(var i = 0; i < len; i++){
            var file = this.files[i]
            if('name' in file)
                alert(file.name)
            else
                alert(file.fileName)
        }

        var file = this.files[0];
        filename = '';
        if('name' in file)
          filename= file.name;
        else
          filename = file.fileName;
        */

        xhr = new XMLHttpRequest();

        (xhr.upload || xhr).addEventListener('progress', function(e) {
            var done = e.position || e.loaded
            var total = e.totalSize || e.total;
            console.log('xhr progress: ' + Math.round(done/total*100) + '%');
            //document.getElementById("load").innerHTML = "Cargando... " + Math.round(done/total*100) + '%';
            document.getElementById("arrastrar").innerHTML = "Cargando... " + Math.round(done/total*100) + '%';
            if(Math.round(done/total*100) == 100){
                //document.getElementById("load").innerHTML = "Cargado: 100%";
                document.getElementById("arrastrar").innerHTML = "Cargado: 100%";
                window.location.replace("procesar_Imagenes.html")
            }
        });
        
        xhr.addEventListener('load', function(e) {
            data = JSON.parse(this.responseText);
            //alert("DATA:" + data)
            if(this.status != 200){
              $('#driver-uploader-failure-alert').show();
              return;
            }
            $('#driver_source').val('Flat file on idisas');
            $('#driver_name').val(data['file_remote_path']).blur();
            $('#driver-uploader-success-alert').show();
        });

        
        fd = new FormData();

        var len = this.files.length
        var fileName_ID = "fileName_"
        var fileName;
        for(var i = 0; i < len; i++){
            var file = this.files[i]
            if('name' in file){
                fileName = file.name
            }else{
                fileName = file.fileName
            }   
            //alert("FileName:", fileName)
            fnName = fileName_ID + i.toString()
            fd.append(fnName, fileName);
            fd.append(fnName, file);
        }
        //xhr.send(fd);
    });
});


$(document).ready(function(){
    $('#upload_data').click(function(){
        xhr.open('post', '/SegmentTrackerAPP/files', true);
        xhr.send(fd);
        //window.location.replace("index.html")
    });
});