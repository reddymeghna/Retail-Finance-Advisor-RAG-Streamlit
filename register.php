<?php include('server.php') ?>
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Registration System PHP and MySQL</title>
  <link rel="stylesheet" type="text/css" href="style.css">
  <style>
    body {
      background-image: url('https://cdn5.vectorstock.com/i/1000x1000/81/24/white-and-gray-color-background-vector-24628124.jpg'); /* Replace with your background image path */
      background-size: cover;
      background-position: center;
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 0;
	  width: 100vw;
    }


    .header {
      background-color: #333;
      color: white;
      text-align: center;
      padding: 10px;
      margin-bottom: 20px;
      border-radius: 8px 8px 0 0; /* Rounded corners for the top */
    }

    .header h2 {
      margin: 0;
    }

    .input-group {
      margin-bottom: 15px;
    }

    label {
      display: block;
      margin-bottom: 8px;
    }

    input[type="text"], input[type="email"], input[type="password"] {
      width: 100%;
      padding: 10px;
      margin-top: 5px;
      margin-bottom: 15px;
      border: 1px solid #ccc;
      border-radius: 4px;
      box-sizing: border-box;
    }

    .btn {
      background-color: #4CAF50;
      color: white;
      padding: 12px 20px;
      border: none;
      cursor: pointer;
      border-radius: 5px;
      text-decoration: none;
      display: block;
      width: 100%;
      font-size: 16px;
      text-align: center;
    }

    .btn:hover {
      background-color: #45a049;
    }

    p {
      margin-top: 15px;
      text-align: center;
    }

    p a {
      color: #333;
      text-decoration: none;
      font-weight: bold;
    }

    p a:hover {
      text-decoration: underline;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h2>Register</h2>
    </div>
      
    <form method="post" action="register.php">
      <?php include('errors.php'); ?>
      <div class="input-group">
        <label>Username</label>
        <input type="text" name="username" value="<?php echo $username; ?>">
      </div>
      <div class="input-group">
        <label>Email</label>
        <input type="email" name="email" value="<?php echo $email; ?>">
      </div>
      <div class="input-group">
        <label>Password</label>
        <input type="password" name="password_1">
      </div>
      <div class="input-group">
        <label>Confirm password</label>
        <input type="password" name="password_2">
      </div>
      <div class="input-group">
        <button type="submit" class="btn" name="reg_user">Register</button>
      </div>
      <p>
        Already a member? <a href="login.php">Sign in</a>
      </p>
    </form>
  </div>
</body>
</html>
