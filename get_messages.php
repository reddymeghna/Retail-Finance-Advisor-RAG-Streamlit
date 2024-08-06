<?php
include 'db_connect.php';

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $user_id = $_POST['user_id'];

    $sql = "SELECT message, timestamp FROM chat_history WHERE user_id = '$user_id' ORDER BY timestamp ASC";
    $result = $conn->query($sql);

    $messages = array();

    if ($result->num_rows > 0) {
        while($row = $result->fetch_assoc()) {
            $messages[] = array('message' => $row["message"], 'timestamp' => $row["timestamp"]);
        }
    }

    echo json_encode($messages);
}

$conn->close();
?>
