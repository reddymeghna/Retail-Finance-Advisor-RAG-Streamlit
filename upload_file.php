<?php
// Check if the form was submitted
if ($_SERVER['REQUEST_METHOD'] == 'POST') {

    // Check if the fileToUpload array key exists
    if (isset($_FILES["fileToUpload"])) {

        $target_dir = "uploads/";
        $target_file = $target_dir . basename($_FILES["fileToUpload"]["name"]);
        $uploadOk = 1;
        $fileType = strtolower(pathinfo($target_file, PATHINFO_EXTENSION));

        // Check if file is a CSV or Excel file
        if ($fileType != "csv" && $fileType != "xlsx") {
            echo "Sorry, only CSV & XLSX files are allowed.";
            $uploadOk = 0;
        }

        // Check file size (example: max 5MB)
        if ($_FILES["fileToUpload"]["size"] > 5000000) {
            echo "Sorry, your file is too large.";
            $uploadOk = 0;
        }

        // Check if $uploadOk is set to 0 by an error
        if ($uploadOk == 0) {
            echo "Sorry, your file was not uploaded.";
        // If everything is ok, try to upload file
        } else {
            if (move_uploaded_file($_FILES["fileToUpload"]["tmp_name"], $target_file)) {
                echo "The file " . htmlspecialchars(basename($_FILES["fileToUpload"]["name"])) . " has been uploaded.";

                // Save file details to the database
                $servername = "localhost";
                $username = "root";
                $password = "";
                $dbname = "streamlit_db";

                // Create connection
                $conn = new mysqli($servername, $username, $password, $dbname);

                // Check connection
                if ($conn->connect_error) {
                    die("Connection failed: " . $conn->connect_error);
                }

                // Use prepared statements to avoid SQL injection
                $stmt = $conn->prepare("INSERT INTO files (filename, file_path) VALUES (?, ?)");
                if ($stmt === false) {
                    die("Prepare failed: " . $conn->error);
                }

                $filename = basename($_FILES["fileToUpload"]["name"]);
                $file_path = $target_file;

                if ($stmt->bind_param("ss", $filename, $file_path) === false) {
                    die("Bind failed: " . $stmt->error);
                }

                if ($stmt->execute()) {
                    echo "File details saved to database.";
                } else {
                    echo "Error: " . $stmt->error; // Display detailed error message
                }

                $stmt->close();
                $conn->close();
            } else {
                echo "Sorry, there was an error uploading your file.";
            }
        }
    } else {
        echo "No file was uploaded.";
    }
} else {
    echo "Invalid request method.";
}
?>
