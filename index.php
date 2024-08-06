<?php
session_start();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Homepage</title>
    <link rel="stylesheet" href="styles.css">
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: hsl(50, 7%, 82%);
            margin: 0;
            padding: 0;
        }

        .navbar {
            font-family: italic;
            overflow: hidden;
            position: absolute; /* Position absolutely within the hero */
            top: 10px; /* Adjust top position */
            right: 20px; /* Adjust right position */
            z-index: 1000; /* Ensure it's above other content */
            text-align: right; /* Align links to the right */
            padding: 10px; /* Adjust padding as needed */
        }

        .navbar-btn {
            display: inline-block;
            background-color: light gray;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-size: 25px;
            padding: 10px 20px;
            margin-left: 10px; /* Adjust spacing between buttons */
            transition: background-color 0.3s ease;
        }

        .navbar-btn:hover {
            background-color: #2980b9;
        }

        .hero {
            background-image: url('https://www.rotageek.com/hubfs/retail-technology-platforms-online-payments.jpg');
            background-size: cover;
            background-position: center;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
            color: white;
            position: relative; /* Ensure hero content is on top */
        }

        .hero-content {
            max-width: 800px;
            padding: 20px;
            position: relative; /* Ensure hero content is on top */
            z-index: 1; /* Ensure hero content is on top */
        }

        .hero-content h1 {
            font-size: 3em;
            margin-bottom: 20px;
        }

        .hero-content p {
            font-size: 1.2em;
            margin-bottom: 20px;
        }

        .btn {
            display: inline-block;
            padding: 10px 20px;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-size: 18px;
            transition: background-color 0.3s ease;
            margin: 5px;
        }

        .btn:hover {
            background-color: #2980b9;
        }

        .login-form {
            background: rgba(0, 0, 0, 0.5);
            padding: 20px;
            border-radius: 5px;
        }

        .login-form input {
            margin: 10px 0;
            padding: 10px;
            width: 100%;
            border: none;
            border-radius: 5px;
        }

        .login-form button {
            padding: 10px 20px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 18px;
            transition: background-color 0.3s ease;
        }

        .login-form button:hover {
            background-color: #2980b9;
        }
    </style>
</head>
<body>
    <!-- Hero section with integrated navigation links -->
    <div class="hero">
        <!-- Navigation links in top right corner -->
        <div class="navbar">
            <a href="#" class="navbar-btn">Home</a>
            <a href="#" class="navbar-btn">About Us</a>
            <a href="#" class="navbar-btn">Contact</a>
        </div>
        
        <!-- Hero content -->
        <div class="hero-content">
            <h1>Welcome to Your Website</h1>
            <p>Explore and engage with our services.</p>
            <a href="login.php" class="btn">Login</a>
        </div>
    </div>
</body>
</html>
