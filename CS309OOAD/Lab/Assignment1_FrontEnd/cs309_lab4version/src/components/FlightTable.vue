<template>
  <div>
    <h1 id="a">Conference Rooms</h1>
    <el-table id="t" :data="rooms" stripe style="border-radius: 4px; box-shadow: 0 2px 20px rgba(0, 0, 0, .912), 0 0 6px rgba(0, 0, 0, .04); width: 95%;;margin: 0 auto;">
      <el-table-column prop="Room Name" label="Room Name" width="120"/>
      <el-table-column prop="Department" label="Department" width="180"/>
      <el-table-column prop="Type" label="Type" width="100"/>
      <el-table-column prop="Location" label="Location" width="180"/>
      <el-table-column prop="Date" label="Date"/>
      <el-table-column prop="Start Time" label="Start Time"/>
      <el-table-column prop="End Time" label="End Time"/>
      <el-table-column prop="Max Duration" label="Max Duration"/>
      <el-table-column label="Operations">
        <template #default="scope">
          <el-button icon="el-icon-delete" id="delete" type="info" @click="deleteRoom(scope.$index)">Delete</el-button>
          <el-button icon="el-icon-edit" id="edit" type="info" @click="editRoom(scope.$index)">Edit</el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-button style="border:transparent; background-color: #909399; margin-left: 45%; margin-top: 20px" icon="el-icon-plus" type="primary" @click="createNewRoom">Add New Room
    </el-button>


    <el-dialog
      :visible.sync="dialogVisible"
      title="Add Room"
      width="60%"
    >
      <el-form
        ref="RoomForm"
        :model="RoomForm"
        :rules="rules"
        label-width="auto"
        label-position="right"
        size="default"
      >
        <el-form-item label="Room Name" prop="RoomName">
          <el-input v-model="RoomForm.RoomName"/>
        </el-form-item>
        <el-form-item label="Department" prop="Department">
          <el-input v-model="RoomForm.Department"/>
        </el-form-item>
        <el-form-item label="Type" prop="Type">
          <el-radio-group v-model="RoomForm.Type">
            <el-radio label="Small">Small</el-radio>
            <el-radio label="Medium">Medium</el-radio>
            <el-radio label="Big">Big</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-row>
          <el-col :span="12">
            <el-form-item label="Location" prop="Location">
              <el-select v-model="RoomForm.Location" clearable placeholder="Building">
                <el-option
                  v-for="item in buildingOptions"
                  :key="item.value"
                  :label="item.label"
                  :value="item.value"
                />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="" prop="roomNo">
              <el-input v-model="RoomForm.roomNo" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item label="Date" prop="Date">
          <el-date-picker
            value-format='yyyy/MM/dd'
            format="yyyy/MM/dd"
            v-model="RoomForm.Date"
            type="date"
            label="Pick a date"
            placeholder="Pick a date"
            style="width: 100%"
          />
        </el-form-item>
        <el-form-item label="Time Range" prop="TimeRange">
          <el-time-picker
            value-format="HH:mm"
            format="HH:mm"
            is-range
            range-separator="To"
            start-placeholder="Start time"
            end-placeholder="End time"
            v-model="RoomForm.TimeRange"
            style="width: 100%"
          />
        </el-form-item>
        <el-form-item label="Max Duration" prop="MaxDuration">
          <el-input v-model="RoomForm.MaxDuration"/>
        </el-form-item>
        <el-form-item>
          <el-button id="edit" icon="el-icon-check" type="primary" @click="AddRoom('RoomForm')">Create</el-button>
        </el-form-item>

      </el-form>
    </el-dialog>
  </div>
</template>

<script>
export default {
  name: 'RoomTable',
  data() {
    const roomNameValidator = (rule, value, callback) => {
      const re = /^[A-Za-z ]+[0-9]+$/;
      if (!value) {
        return callback(new Error('Please input Room Name'));
      }
      if (!re.test(value)) {
        return callback(new Error('Invalid Room Name'));
      }
      callback();
    };

    const departmentValidator = (rule, value, callback) => {
      const re = /^[A-Z a-z]+$/;
      if (!value) {
        callback(new Error('Please input Department Name'));
      }
      if (!re.test(value)) {
        return callback(new Error('Invalid Department Name'));
      }
      callback();
    };
    const roomNoValidator = (rule, value, callback) => {
      const re = /^[A-Za-z ]+[0-9]+$/;
      if (!value) {
        callback(new Error('Please input Room No'));
      }
      if (!re.test(value)) {
        return callback(new Error('Invalid Room No'));
      }
      callback();
    };
    const durationValidator = (rule, value, callback) => {
      const re = /^[0-9]+[h]$/;
      if (!value) {
        callback(new Error('Please input Max Duration'));
      }
      if (!re.test(value)) {
        return callback(new Error('Invalid Max Duration'));
      }
      callback();
    };
    const dateValidator = (rule, value, callback) => {
      const currentDate = new Date();
      const selectedDate = new Date(value);
      if (!value) {
        callback(new Error('Please input Date'));
      }
      if (selectedDate < currentDate) {
        return callback(new Error('Invalid Date'));
      }
      callback();
    };
    return {
      rooms: [
        {
          "Room Name": "Room1",
          "Department": "Electrical",
          "Type": "Small",
          "Location": "South Building 426A",
          "Date": "2023/09/10",
          "Start Time": "08:00",
          "End Time": "20:00",
          "Max Duration": "2h"
        },
        {
          "Room Name": "Room2",
          "Department": "Computer Science",
          "Type": "Big",
          "Location": "South Building 434A",
          "Date": "2023/09/10",
          "Start Time": "00:00",
          "End Time": "24:00",
          "Max Duration": "4h"
        },
      ],
      RoomForm: {
        RoomName: "",
        Department: "",
        Type: "",
        Location: "",
        roomNo: "",
        Date: "",
        TimeRange: "",
        MaxDuration: "",
      },
      buildingOptions: [
        {
          value: 'Teaching Building No.1 Lecture Hall',
          label: 'Teaching Building No.1 Lecture Hall',
        },
        {
          value: 'Research Building Lecture Hall',
          label: 'Research Building Lecture Hall',
        },
        {
          value: 'Library Conference Hall',
          label: 'Library Conference Hall',
        },
        {
          value: 'South Building',
          label: 'South Building',
        },
      ],
      rules: {
        RoomName: [
          {validator: roomNameValidator, trigger: 'blur'},
          {required: true, trigger: true}
        ],
        Type: [
          {required: true, message: 'Please input Room Type', trigger: 'blur'},
        ],
        Department: [
          {validator: departmentValidator, trigger: 'blur'},
          {required: true, trigger: true}
        ],
        Location: [
          {required: true, message: 'Please input Department Name', trigger: true}
        ],
        roomNo: [
          {validator: roomNoValidator, trigger: 'blur'},
          {required: true, trigger: true}
        ],
        Date: [
          {validator: dateValidator, trigger: 'blur'},
          {required: true, message: 'Please select the date', trigger: 'blur'},
        ],
        TimeRange: [
          {required: true, message: 'Please select the time range', trigger: 'blur'},
        ],
        MaxDuration: [
          {validator: durationValidator, trigger: 'blur'},
          {required: true, trigger: true}
        ]
      },
      dialogVisible: false,
      addOrNot: true,
      editIndex: -1,
    }
  },
  methods: {
    createNewRoom() {
      // console.log(this.dialogVisible)
      this.dialogVisible = true
      this.addOrNot = true
      // console.log(this.dialogVisible)
      this.$refs["RoomForm"].resetFields();
      this.RoomForm= {
        RoomName: "",
        Department: "",
        Type: "",
        Location: "",
        roomNo: "",
        Date: "",
        TimeRange: "",
        MaxDuration: "",
      }
    },
    deleteRoom(index) {
      this.rooms.splice(index, 1)
    },
    editRoom(index) {
      // load original info
      this.dialogVisible = true
      this.RoomForm.RoomName = this.rooms[index]["Room Name"]
      this.RoomForm.Department = this.rooms[index]["Department"]
      this.RoomForm.Type = this.rooms[index]["Type"]
      let temp = this.rooms[index]["Location"].lastIndexOf(" ")
      this.RoomForm.Location = this.rooms[index]["Location"].substring(0, temp)
      this.RoomForm.roomNo = this.rooms[index]["Location"].substring(temp+1)
      this.RoomForm.Date = this.rooms[index]["Date"]
      this.RoomForm.TimeRange = [this.rooms[index]["Start Time"], this.rooms[index]["End Time"]]
      // this.RoomForm.EndTime = this.rooms[index]["End Time"]
      this.RoomForm.MaxDuration = this.rooms[index]["Max Duration"]
      this.addOrNot = false
      this.editIndex = index
    },
    AddRoom(formName) {
      console.log(this.RoomForm.Date)
      this.$refs[formName].validate((valid) => {
        if (valid) {
          if (this.addOrNot){
            this.rooms.push({
              "Room Name": this.RoomForm.RoomName,
              "Department": this.RoomForm.Department,
              "Type": this.RoomForm.Type,
              "Location": this.RoomForm.Location + " " + this.RoomForm.roomNo,
              "Date": this.RoomForm.Date,
              "Start Time": this.RoomForm.TimeRange[0],
              "End Time": this.RoomForm.TimeRange[1],
              "Max Duration": this.RoomForm.MaxDuration
            })
            alert('submit!')
            this.dialogVisible = false
          }else{
            this.rooms.splice(this.editIndex, 1, {
              "Room Name": this.RoomForm.RoomName,
              "Department": this.RoomForm.Department,
              "Type": this.RoomForm.Type,
              "Location": this.RoomForm.Location + " " + this.RoomForm.roomNo,
              "Date": this.RoomForm.Date,
              "Start Time": this.RoomForm.TimeRange[0],
              "End Time": this.RoomForm.TimeRange[1],
              "Max Duration": this.RoomForm.MaxDuration
            })
            alert('edited!')
            this.dialogVisible = false
            this.addOrNot = true
          }
        } else {
          console.log("Add Fail");
        }
      })
    }
  }

}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
#a {
  text-align: center;
  font-size: 40px;
  margin: 20px;
  text-shadow:
    1px 1px 0 #eebe77,
    -1px -1px 0 #eebe77,
    1px -1px 0 #eebe77,
    -1px 1px 0 #eebe77;
}
#edit {
  margin: 5px;
  width: 100%;
  background-color: #eebe77;
  border: transparent;
}
#delete {
  width: 100%;
  margin: 5px;
}

</style>

<style>
body{
  background-image: url("../assets/meeting_room.jpg");
  background-size: cover;
  background-position: center center;
  background-repeat: no-repeat;
  background-attachment: fixed;
}
</style>
