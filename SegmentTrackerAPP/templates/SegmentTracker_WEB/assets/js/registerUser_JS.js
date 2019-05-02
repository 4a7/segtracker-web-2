
function encrypt(pass){
    var key = "segtrackerpasskey";

    // PROCESS
    var encrypted = pass //CryptoJS.AES.encrypt(pass, key);
    return encrypted
    //var decrypted = CryptoJS.AES.decrypt(encrypted, key);
    //alert(pass);
    //alert(encrypted);
    //alert(decrypted);
    //alert(decrypted.toString(CryptoJS.enc.Utf8));

}

//VERIFICA QUE EL USUARIO TENGA LOS DATOS CORRECTOS Y MANDA LA INFORMACIÓN AL SERVIDOR PARA SER AGREGADO A LA BD
function addUser(){
    n = document.getElementById("name").value;
    l = document.getElementById("lastname").value;
    e = document.getElementById("email").value;
    i = document.getElementById("institute").value;
    p = document.getElementById("pass").value;
    rp = document.getElementById("repass").value;

    //alert("Info: " + "," + n + "," + e + "," + i + "," + p + "," + rp)

    var name_regex = /[[A-ZÀ-ÿa-z]+[[A-ZÀ-ÿa-z ]*/;
    var email_regex = /^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,4})+$/;
    var institute_regex = /^[A-ZÀ-ÿa-z0-9,.(\%\2\0)* ]+$/;
    var password_regex = /^([A-Za-z0-9]{6,15})+$/;
    var ok = true;
    
    if(!name_regex.test(n)){
        alert("Por favor ingrese su nombre.")
        ok = false
        return
    }

    if(!name_regex.test(l)){
        alert("Por favor ingrese su apellido.")
        ok = false
        return
    }

    if(!email_regex.test(e)){
        alert("Por favor ingrese un correo electrónico válido.")   
        ok = false
        return     
    }

    if(!institute_regex.test(i)){
        alert("Por favor ingrese el nombre de la institución al que pertenece.")
        ok = false
        return
    }

    if(p==rp){
        if(!password_regex.test(p)){
            alert("La contraseña debe contener únicamente letras y números con una extensión de 6 a 15 caracteres.")
            ok = false
            return
        }
    }else{
        alert("Ambas contraseñas ingresadas deben ser iguales")
        ok = false
        return
    }

    if(ok == true){
        p = encrypt(p)
        makeRequest("registrate/?name="+n+"&lastname="+l+"&email="+e+"&institute="+i+"&HsyRdIC0coC4rJu4="+p)
        var present = xmlHttpRequest.responseText
        if(present == "YES"){
            alert("El usuario ya se encuentra registrado.")
        }else{
            alert("Usuario registrado con éxito")
            window.location.replace("index.html")
        }
    }else{
        alert("No se pudo registrar al usuario.")
        return
    }
}


