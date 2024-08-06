<?php

include('server.php');



// Initialize variables
$username = "";
$errors = array(); 

// LOGIN USER
if (isset($_POST['login_user'])) {
    $username = mysqli_real_escape_string($db, $_POST['username']);
    $password = mysqli_real_escape_string($db, $_POST['password']);

    if (empty($username)) {
        array_push($errors, "Username is required");
    }
    if (empty($password)) {
        array_push($errors, "Password is required");
    }

    if (count($errors) == 0) {
        $password = md5($password); // Assuming you have hashed the passwords with md5
        $query = "SELECT * FROM girls WHERE username='$username' AND password='$password'";
        $results = mysqli_query($db, $query);

        if (mysqli_num_rows($results) == 1) {
            $_SESSION['username'] = $username;
            $_SESSION['success'] = "You are now logged in";
            start_streamlit_server();
            header('Location: http://localhost:8501'); // Redirect to the Streamlit app
            exit();
        } else {
            array_push($errors, "Wrong username/password combination");
        }
    }
}
?>
<!DOCTYPE html>
<html>
<head>
    <title>Login Page</title>
    <link rel="stylesheet" type="text/css" href="style.css">
    <style>
        body {
            background-image: url('https://images.vexels.com/media/users/3/214568/raw/fbed067746e420da0cc755da23090a96-honeycomb-white-background-design.jpg'); 
            background-size: cover;
            background-color: gray;
            background-position: center;
            font-family: Verdana, sans-serif;
            margin: 50;
            padding: 0;
        }

        .header {
            background-color: black;
            color: white;
            text-align: center;
            padding: 20px;
        }

        .input-group {
            margin: 50px;
        }

        label {
            display: block;
            margin-bottom: 10px;
        }

        input[type="text"], input[type="password"] {
            width: 100%;
            height: 10vh;
            padding: 8px;
            margin: 5px 0 15px 0;
            border: 1px solid gray;
            border-radius: 10px;
            box-sizing: border-box;
        }

        .btn {
            background-color: #4CAF50;
            color: black;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            border-radius: 5px;
            text-decoration: none;
        }

        .btn:hover {
            background-color: #45a049;
        }

        p {
            margin-top: 25px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h2>Login</h2>
    </div>
     
    <form method="post" action="login.php" autocomplete="off">
        <?php include('errors.php'); ?>
        <div class="input-group">
            <label>Username</label>
            <input type="text" name="username" autocomplete="off">
        </div>
        <div class="input-group">
            <label>Password</label>
            <input type="password" name="password" autocomplete="new-password">
        </div>
        <div class="input-group">
            <button type="submit" class="btn" name="login_user">Login</button>
        </div>
        <p>
            Not yet a member? <a href="register.php">Sign up</a>
        </p>
        <p>
            Forgot your password? <a href="reset_password.php">Reset it</a>
        </p>
    </form>
</body>
</html>
