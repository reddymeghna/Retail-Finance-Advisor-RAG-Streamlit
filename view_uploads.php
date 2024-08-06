<?php
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

// Fetch file details from the database
$sql = "SELECT id, filename, file_path FROM files";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
    echo "<table border='1'>";
    echo "<tr><th>ID</th><th>Filename</th><th>File Path</th><th>Download</th></tr>";
    
    // Output data of each row
    while($row = $result->fetch_assoc()) {
        echo "<tr>";
        echo "<td>" . htmlspecialchars($row["id"]) . "</td>";
        echo "<td>" . htmlspecialchars($row["filename"]) . "</td>";
        echo "<td>" . htmlspecialchars($row["file_path"]) . "</td>";
        echo "<td><a href='" . htmlspecialchars($row["file_path"]) . "' download>Download</a></td>";
        echo "</tr>";
    }
    echo "</table>";
} else {
    echo "0 results";
}

$conn->close();
?>
