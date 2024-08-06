<?php
session_start();

$isLogin = true;
$email = $password = $username = "";
$errors = [];

if ($_SERVER["REQUEST_METHOD"] === "POST") {
    if (isset($_POST['formType'])) {
        $isLogin = $_POST['formType'] === 'login';
    }

    if (!$isLogin) {
        $username = trim($_POST['username']);
        if (empty($username)) {
            $errors[] = "Username is required.";
        }
    }

    $email = trim($_POST['email']);
    $password = trim($_POST['password']);

    if (empty($email)) {
        $errors[] = "Email is required.";
    }

    if (empty($password)) {
        $errors[] = "Password is required.";
    }

    if (empty($errors)) {
        if ($isLogin) {
            // Handle login logic here
            echo "<script>alert('Logging in: Email: $email, Password: $password');</script>";
        } else {
            // Handle registration logic here
            echo "<script>alert('Registering: Username: $username, Email: $email, Password: $password');</script>";
        }
    }
}

function input_value($field_name, $isLogin) {
    if ($field_name == "username" && !$isLogin) {
        return $_POST['username'] ?? '';
    }
    return $_POST[$field_name] ?? '';
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login / Register</title>
    <link rel="stylesheet" href="login.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
</head>
<body>
    <div class="container">
        <div class="box">
            <div class="top-header">
                <h3><?php echo $isLogin ? "Hello, Again!" : "Sign Up, Now!"; ?></h3>
                <small>We are happy to have you <?php echo $isLogin ? "back" : "with us"; ?>.</small>
            </div>
            <?php if (!empty($errors)): ?>
                <ul class="errors">
                    <?php foreach ($errors as $error): ?>
                        <li><?php echo $error; ?></li>
                    <?php endforeach; ?>
                </ul>
            <?php endif; ?>
            <form method="post" action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]); ?>">
                <input type="hidden" name="formType" value="<?php echo $isLogin ? 'login' : 'register'; ?>">
                <div class="input-group">
                    <?php if (!$isLogin): ?>
                        <div class="input-field">
                            <input
                                type="text"
                                class="input-box"
                                id="regUsername"
                                name="username"
                                value="<?php echo input_value('username', $isLogin); ?>"
                                required
                            />
                            <label for="regUsername">Username</label>
                        </div>
                    <?php endif; ?>
                    <div class="input-field">
                        <input
                            type="text"
                            class="input-box"
                            id="logEmail"
                            name="email"
                            value="<?php echo input_value('email', $isLogin); ?>"
                            required
                        />
                        <label for="logEmail">Email address</label>
                    </div>
                    <div class="input-field">
                        <input
                            type="password"
                            class="input-box"
                            id="<?php echo $isLogin ? 'logPassword' : 'regPassword'; ?>"
                            name="password"
                            value="<?php echo input_value('password', $isLogin); ?>"
                            required
                        />
                        <label for="<?php echo $isLogin ? 'logPassword' : 'regPassword'; ?>">Password</label>
                        <div class="eye-area">
                            <div class="eye-box" onclick="togglePasswordVisibility()">
                                <i id="passwordToggleIcon" class="fa-regular fa-eye-slash"></i>
                            </div>
                        </div>
                    </div>
                    <div class="remember">
                        <input type="checkbox" id="<?php echo $isLogin ? 'formCheck' : 'formCheck2'; ?>" class="check" />
                        <label for="<?php echo $isLogin ? 'formCheck' : 'formCheck2'; ?>">Remember Me</label>
                    </div>
                    <div class="input-field">
                        <input type="submit" class="input-submit" value="<?php echo $isLogin ? 'Sign In' : 'Sign Up'; ?>" />
                    </div>
                    <?php if ($isLogin): ?>
                        <div class="forgot">
                            <a href="#">Forgot password?</a>
                        </div>
                    <?php endif; ?>
                </div>
            </form>
            <div class="switch">
                <a href="#" class="login <?php echo $isLogin ? 'active' : ''; ?>" onclick="switchForm(true)">Login</a>
                <a href="#" class="register <?php echo !$isLogin ? 'active' : ''; ?>" onclick="switchForm(false)">Register</a>
                <div class="btn-active" id="btn" style="left: <?php echo $isLogin ? '0px' : '150px'; ?>"></div>
            </div>
        </div>
    </div>
    <script>
        function togglePasswordVisibility() {
            const passwordInput = document.querySelector('.input-box[type="password"]');
            const passwordIcon = document.getElementById('passwordToggleIcon');

            if (passwordInput.type === 'password') {
                passwordInput.type = 'text';
                passwordIcon.classList.remove('fa-eye-slash');
                passwordIcon.classList.add('fa-eye');
            } else {
                passwordInput.type = 'password';
                passwordIcon.classList.remove('fa-eye');
                passwordIcon.classList.add('fa-eye-slash');
            }
        }

        function switchForm(login) {
            const formType = document.querySelector('input[name="formType"]');
            formType.value = login ? 'login' : 'register';
            const form = document.querySelector('form');
            form.submit();
        }
    </script>
</body>
</html>
