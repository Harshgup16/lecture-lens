<<<<<<< HEAD
document.addEventListener("DOMContentLoaded", () => {
    const activePage = window.location.pathname; // Get the current page path
    const navLinks = document.querySelectorAll('#action_bar a'); // Select only nav links

    navLinks.forEach(link => {
        // Check if the link's href includes the active page
        if (link.href.includes(activePage)) {
            link.classList.add('active'); // Add 'active' class to the current link
        }
    });
});
=======
document.addEventListener("DOMContentLoaded", () => {
    const activePage = window.location.pathname; // Get the current page path
    const navLinks = document.querySelectorAll('#action_bar a'); // Select only nav links

    navLinks.forEach(link => {
        // Check if the link's href includes the active page
        if (link.href.includes(activePage)) {
            link.classList.add('active'); // Add 'active' class to the current link
        }
    });
});
>>>>>>> 698f0f17fb6c5a18403414d76267ca4ec99c7bcc
