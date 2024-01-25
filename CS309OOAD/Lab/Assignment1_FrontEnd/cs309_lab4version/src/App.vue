<template>
  <RoomTable />
</template>

<script>
import RoomTable from './components/FlightTable.vue'

export default {
  name: 'App',
  components: {
    RoomTable
  },
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
