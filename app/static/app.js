const api = {
  get: async (url) => {
    const resp = await fetch(url, { credentials: "include" });
    if (resp.status === 401) {
      window.location.href = "/";
      return null;
    }
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Request failed: ${resp.status} ${text}`);
    }
    return resp.json();
  },
};

async function loadDashboard() {
  const statusEl = document.querySelector("#status");
  try {
    const [me, stats, activities] = await Promise.all([
      api.get("/api/me"),
      api.get("/api/stats"),
      api.get("/api/activities?per_page=10"),
    ]);

    if (!me) {
      return;
    }

    document.querySelector("#welcome").textContent = `Hi ${me.username || me.athlete_id}!`;

    if (stats) {
      const totals = stats.totals;
      document.querySelector("#total-distance").textContent = `${totals.distance_km ?? 0} km`;
      document.querySelector("#total-time").textContent = `${totals.moving_time_hours ?? 0} h`;
      document.querySelector("#total-elev").textContent = `${totals.elev_gain_m ?? 0} m`;
      document.querySelector("#total-pace").textContent = totals.avg_pace_sec_per_km
        ? `${(totals.avg_pace_sec_per_km / 60).toFixed(2)} min/km`
        : "n/a";
    }

    const list = document.querySelector("#activity-list");
    list.innerHTML = "";
    (activities || []).forEach((activity) => {
      const li = document.createElement("li");
      const distance = ((activity.distance || 0) / 1000).toFixed(2);
      const moving = ((activity.moving_time || 0) / 60).toFixed(1);
      li.textContent = `${activity.name} ? ${distance} km in ${moving} min`;
      list.appendChild(li);
    });

    statusEl.textContent = "Latest Strava data loaded.";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Failed to load dashboard data.";
  }
}

async function logout() {
  await fetch("/api/auth/logout", { method: "POST", credentials: "include" });
  window.location.href = "/";
}

document.addEventListener("DOMContentLoaded", () => {
  const logoutButton = document.querySelector("#logout");
  if (logoutButton) {
    logoutButton.addEventListener("click", logout);
  }
  loadDashboard();
});
